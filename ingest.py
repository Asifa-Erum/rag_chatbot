import os
import re
import uuid
import requests
import json
from pathlib import Path
from bs4 import BeautifulSoup
from lxml import etree
import hashlib # Added for deterministic IDs
from langchain_text_splitters import RecursiveCharacterTextSplitter # Corrected import
from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings

# --- Configuration ---
def load_config():
    """Loads configuration from config.json"""
    try:
        script_dir = Path(__file__).parent
        config_path = script_dir / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Ensure default collection name is 'physical_ai_book'
        config.setdefault("QDRANT_COLLECTION_NAME", "physical_ai_book")
        return config
    except FileNotFoundError:
        print("FATAL: config.json not found. Please copy config.json.example to config.json and fill in your credentials.")
        exit(1)
    except json.JSONDecodeError:
        print("FATAL: Could not decode config.json. Please ensure it is a valid JSON file.")
        exit(1)

config = load_config()

SITEMAP_PATH = "../handbook-site/build/sitemap.xml"
QDRANT_COLLECTION_NAME = config["QDRANT_COLLECTION_NAME"]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- API Clients ---
try:
    qdrant_client = QdrantClient(
        url=config["QDRANT_URL"],
        api_key=config["QDRANT_API_KEY"],
        timeout=60 # Increased timeout for potentially large upsert operations
    )
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=config["OPENAI_API_KEY"]
    )
except KeyError as e:
    print(f"FATAL: Missing required key in config.json: {e}")
    exit(1)
except Exception as e:
    print(f"Error initializing clients: {e}")
    exit(1)

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ... (rest of the imports and existing code) ...

# --- Helper Functions ---
def setup_qdrant_collection():
    """Ensures the Qdrant collection is ready, creating or recreating it."""
    print(f"Setting up Qdrant collection '{QDRANT_COLLECTION_NAME}'...")
    try:
        # Always recreate the collection to ensure a clean state for fresh ingestion
        qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1536,  # OpenAI text-embedding-3-small
                distance=models.Distance.COSINE
            ),
        )
        print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created/recreated.")
    except Exception as e:
        print(f"An error occurred with Qdrant collection setup: {e}")
        exit(1) # Exit if collection setup fails

def get_urls_from_sitemap(sitemap_path: str) -> list[str]:
    """Reads a local sitemap file and extracts all URLs."""
    print(f"Reading sitemap from: {sitemap_path}")
    try:
        with open(sitemap_path, 'r', encoding='utf-8') as f:
            sitemap_content = f.read()
        
        root = etree.fromstring(sitemap_content.encode('utf-8'))
        namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [url.text for url in root.findall('sitemap:url/sitemap:loc', namespace)]
        
        print(f"Found {len(urls)} URLs in the sitemap.")
        return urls
    except FileNotFoundError:
        print(f"Error: Sitemap file not found at {sitemap_path}")
        return []
    except Exception as e:
        print(f"Error reading sitemap: {e}")
        return []

def scrape_page_content(url: str) -> str:
    """Reads the content from a local file path derived from a URL."""
    # Convert the URL to a local file path
    # The base URL from the sitemap needs to be replaced by the local build directory
    base_url = "https://physical-ai-book-one.vercel.app"
    local_base_path = Path(__file__).parent.parent / "handbook-site/build"
    
    # Create a relative path from the URL
    if url.startswith(base_url):
        relative_path = url[len(base_url):]
    else:
        print(f"URL {url} does not match the expected base URL.")
        return ""

    # Remove leading slash
    if relative_path.startswith('/'):
        relative_path = relative_path[1:]

    file_path = local_base_path / relative_path

    # If the path is a directory, append index.html
    if file_path.is_dir():
        file_path = file_path / "index.html"
    
    print(f"Reading content from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            main_content = soup.find('main')
            if main_content:
                return main_content.get_text(separator=' ', strip=True)
            else:
                return soup.get_text(separator=' ', strip=True)
    except FileNotFoundError:
        print(f"Could not find file: {file_path}")
        return ""
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return ""

def get_documents(urls: list[str]):
    """Crawls a list of URLs and returns the scraped content."""
    print(f"Scraping content from {len(urls)} URLs...")
    documents = []
    for url in urls:
        content = scrape_page_content(url)
        if content:
            documents.append({
                "text": content,
                "source": url
            })
    print(f"Successfully scraped {len(documents)} pages.")
    return documents

def chunk_documents(documents):
    """Splits documents into smaller chunks."""
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc["text"])
        for text in split_texts:
            # Generate deterministic ID
            unique_id_string = f"{doc['source']}-{text}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, unique_id_string)) # Generate deterministic UUID
            chunks.append({
                "id": chunk_id, # Use deterministic ID
                "text": text,
                "source": doc["source"]
            })
    print(f"Created {len(chunks)} text chunks.")
    return chunks

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception), # Retry on any exception for robustness
    reraise=True
)
def store_in_qdrant(chunks):
    """Generates embeddings and stores the chunks in Qdrant with retries."""
    if not chunks:
        print("No chunks to store.")
        return
        
    print(f"Preparing to store {len(chunks)} chunks in Qdrant...")
    
    texts_to_embed = [chunk["text"] for chunk in chunks]
    
    print("Generating embeddings with OpenAI...")
    embedded_texts = embeddings_model.embed_documents(texts_to_embed)
    print("Embeddings generated.")

    points = [
        models.PointStruct(
            id=chunk["id"],
            vector=embedded_texts[i],
            payload={"text": chunk["text"], "source": chunk["source"]}
        )
        for i, chunk in enumerate(chunks)
    ]

    print("Upserting data into Qdrant in batches...")
    BATCH_SIZE = 100 # Define batch size for upsert operations
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        print(f"  Upserting batch {i//BATCH_SIZE + 1}/{(len(points)-1)//BATCH_SIZE + 1} ({len(batch)} points)...")
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=batch,
            wait=True
        )
    print("Data successfully stored in Qdrant.")

# --- Main Execution ---
def main():
    """Main function to run the full ingestion pipeline."""
    print("--- Starting Content Ingestion from Local Files ---")
    setup_qdrant_collection()
    urls = get_urls_from_sitemap(SITEMAP_PATH)
    if urls:
        documents = get_documents(urls)
        if documents:
            chunks = chunk_documents(documents)
            store_in_qdrant(chunks)
    
    print("--- Content Ingestion Complete ---")

if __name__ == "__main__":
    main()