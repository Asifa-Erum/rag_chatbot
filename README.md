# RAG Chatbot Backend for Physical AI Textbook (v3)

This project implements a Retrieval-Augmented Generation (RAG) chatbot that crawls a live website for its knowledge base. It uses a FastAPI backend, Qdrant for vector search, and OpenAI for language understanding and generation.

This version uses a `config.json` file for configuration to ensure reliability.

## How It Works

1.  **Crawl Sitemap**: The ingestion script (`ingest.py`) starts by fetching and parsing the `sitemap.xml` of the online book.
2.  **Scrape Content**: It then visits each URL found in the sitemap and uses BeautifulSoup to scrape the clean, textual content from the main body of each page.
3.  **Chunk & Embed**: The scraped text is split into smaller, manageable chunks. The script then calls the OpenAI API to generate a vector embedding for each chunk.
4.  **Store**: The embeddings and their corresponding text/metadata (like the source URL) are stored in a Qdrant vector database.
5.  **Serve API**: The FastAPI server (`main.py`) exposes a `/api/chat` endpoint that accepts user queries, finds relevant context in Qdrant, and generates a streaming answer using an OpenAI LLM.

## 1. Setup and Configuration

### Prerequisites

*   **Python**: Version 3.9 or higher.
*   **OpenAI Account**: An API key is required.
*   **Qdrant Cloud Account**: A free tier account is sufficient. Get your cluster URL and generate an API key.

### Step 1: Create Your Configuration File

Instead of a `.env` file, this project uses a `config.json` file for credentials.

1.  In the `rag-chatbot-backend` directory, create a `config.json` file by copying the example:
    ```bash
    cp config.json.example config.json
    ```
2.  Open `config.json` and fill in your secret credentials.

    ```json
    {
      "OPENAI_API_KEY": "sk-...",
      "QDRANT_URL": "https://your-qdrant-cluster-url.cloud.qdrant.io",
      "QDRANT_API_KEY": "your-qdrant-api-key",
      "QDRANT_COLLECTION_NAME": "physical_ai_book_v2"
    }
    ```

### Step 2: Install Dependencies

Create and activate a virtual environment, then install the required packages.

```bash
# From within the rag-chatbot-backend directory
python -m venv .venv

# Activate the environment
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

## 2. Run the Data Ingestion

This one-time step crawls the website and populates your Qdrant database. Run it again only if the online book content changes.

```bash
python ingest.py
```

## 3. Run the Backend Server

This command starts the FastAPI server. It should now start without errors.

```bash
python -m uvicorn main:app
```

The server will be available at `http://127.0.0.1:8000`.

## 4. API Usage & Testing

With the server running, open a **new terminal** and use `curl` (or `Invoke-WebRequest` in PowerShell) to test the API.

### PowerShell Example:

```powershell
(Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/chat" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"query": "What is Sim2Real in robotics?"}').Content
```

### Bash/cURL Example:
```bash
curl -X POST "http://127.0.0.1:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d 
{
           "query": "What is Sim2Real in robotics?"
         }
```

---

## Full Code Reference

### `requirements.txt`

```
fastapi
uvicorn[standard]
python-dotenv
openai
qdrant-client
langchain
langchain-openai
langchain-qdrant
tiktoken
requests
beautifulsoup4
lxml
```

### `config.json.example`

```json
{
  "OPENAI_API_KEY": "sk-...",
  "QDRANT_URL": "https://...",
  "QDRANT_API_KEY": "...",
  "QDRANT_COLLECTION_NAME": "physical_ai_book_v2"
}
```

### `ingest.py`

```python
import os
import re
import uuid
import requests
import json
from pathlib import Path
from bs4 import BeautifulSoup
from lxml import etree
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
        config.setdefault("QDRANT_COLLECTION_NAME", "physical_ai_book_v2")
        return config
    except FileNotFoundError:
        print("FATAL: config.json not found. Please copy config.json.example to config.json and fill in your credentials.")
        exit(1)
    except json.JSONDecodeError:
        print("FATAL: Could not decode config.json. Please ensure it is a valid JSON file.")
        exit(1)

config = load_config()

SITEMAP_URL = "https://physical-ai-humanoid-robotics-book-two.vercel.app/sitemap.xml"
QDRANT_COLLECTION_NAME = config["QDRANT_COLLECTION_NAME"]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- API Clients ---
try:
    qdrant_client = QdrantClient(
        url=config["QDRANT_URL"],
        api_key=config["QDRANT_API_KEY"],
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

# --- Helper Functions ---
def setup_qdrant_collection():
    """Ensures the Qdrant collection is ready."""
    print("Setting up Qdrant collection...")
    try:
        qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=1536,
                distance=models.Distance.COSINE
            ),
        )
        print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created/recreated.")
    except Exception as e:
        print(f"Notice: Could not recreate Qdrant collection: {e}")

def get_urls_from_sitemap(sitemap_url: str) -> list[str]:
    """Fetches a sitemap and extracts all URLs."""
    print(f"Fetching sitemap from: {sitemap_url}")
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        root = etree.fromstring(response.content)
        namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [url.text for url in root.findall('sitemap:url/sitemap:loc', namespace)]
        print(f"Found {len(urls)} URLs in the sitemap.")
        return urls
    except requests.RequestException as e:
        print(f"Error fetching sitemap: {e}")
        return []

def scrape_page_content(url: str) -> str:
    """Scrapes the main textual content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find('main')
        return main_content.get_text(separator=' ', strip=True) if main_content else soup.get_text(separator=' ', strip=True)
    except requests.RequestException as e:
        print(f"Could not scrape {url}: {e}")
        return ""

def get_documents(urls: list[str]):
    """Crawls a list of URLs and returns the scraped content."""
    print(f"Scraping content from {len(urls)} URLs...")
    documents = []
    for url in urls:
        content = scrape_page_content(url)
        if content:
            documents.append({"text": content, "source": url})
    print(f"Successfully scraped {len(documents)} pages.")
    return documents

def chunk_documents(documents):
    """Splits documents into smaller chunks."""
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for doc in documents:
        for text in text_splitter.split_text(doc["text"]):
            chunks.append({"id": str(uuid.uuid4()), "text": text, "source": doc["source"]})
    print(f"Created {len(chunks)} text chunks.")
    return chunks

def store_in_qdrant(chunks):
    """Generates embeddings and stores the chunks in Qdrant."""
    if not chunks:
        print("No chunks to store.")
        return
    
    print("Generating embeddings with OpenAI...")
    embedded_texts = embeddings_model.embed_documents([chunk["text"] for chunk in chunks])
    print("Embeddings generated.")

    points = [
        models.PointStruct(id=chunk["id"], vector=embedded_texts[i], payload=chunk)
        for i, chunk in enumerate(chunks)
    ]

    print(f"Upserting {len(points)} points into Qdrant...")
    qdrant_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points, wait=True)
    print("Data successfully stored in Qdrant.")

def main():
    """Main function to run the full ingestion pipeline."""
    print("--- Starting Content Ingestion from Web ---")
    setup_qdrant_collection()
    urls = get_urls_from_sitemap(SITEMAP_URL)
    if urls:
        documents = get_documents(urls)
        if documents:
            chunks = chunk_documents(documents)
            store_in_qdrant(chunks)
    print("--- Content Ingestion Complete ---")

if __name__ == "__main__":
    main()
```

### `main.py`

```python
import os
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from openai import AsyncOpenAI
from fastapi.responses import StreamingResponse
import asyncio

# --- Configuration ---
def load_config():
    """Loads configuration from config.json"""
    try:
        script_dir = Path(__file__).parent
        config_path = script_dir / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        config.setdefault("QDRANT_COLLECTION_NAME", "physical_ai_book_v2")
        return config
    except FileNotFoundError:
        print("FATAL: config.json not found. Please copy config.json.example to config.json and fill in your credentials.")
        exit(1)
    except json.JSONDecodeError:
        print("FATAL: Could not decode config.json. Please ensure it is a valid JSON file.")
        exit(1)

config = load_config()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Physical AI Textbook RAG Chatbot",
    description="A chatbot that answers questions based on the Physical AI & Humanoid Robotics textbook.",
    version="2.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- API Clients ---
try:
    qdrant_client = QdrantClient(url=config["QDRANT_URL"], api_key=config["QDRANT_API_KEY"])
    openai_client = AsyncOpenAI(api_key=config["OPENAI_API_KEY"])
    qdrant_collection_name = config["QDRANT_COLLECTION_NAME"]
except KeyError as e:
    print(f"FATAL: Missing required key in config.json: {e}")
    exit(1)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str
    selected_text: str | None = None

# --- Core RAG Logic ---
async def get_relevant_context(query: str, limit: int = 5) -> list[dict]:
    """Queries Qdrant to find the most relevant text chunks for a given query."""
    try:
        query_embedding_response = await openai_client.embeddings.create(input=[query], model="text-embedding-3-small")
        query_embedding = query_embedding_response.data[0].embedding

        search_results = qdrant_client.search(
            collection_name=qdrant_collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True,
        )
        
        return [point.payload for point in search_results]
    except Exception as e:
        print(f"Error getting context from Qdrant: {e}")
        return []

# --- API Endpoints ---
@app.get("/", summary="Health Check")
def read_root():
    return {"status": "ok"}

@app.post("/api/chat", summary="Handle Chat Queries")
async def chat_handler(request: ChatRequest):
    """Handles chatbot queries with streaming response."""
    context_str = ""
    sources = []

    if request.selected_text:
        context_str = request.selected_text
        sources = ["User-selected text"]
    else:
        relevant_context = await get_relevant_context(request.query)
        if relevant_context:
            context_str = "\n\n---\n\n".join([item['text'] for item in relevant_context])
            sources = sorted(list(set([item['source'] for item in relevant_context])))
        else:
            context_str = "No relevant context was found in the book to answer this question."

    system_prompt = """
    You are an expert assistant for the "Physical AI & Humanoid Robotics" textbook.
    Your role is to answer questions accurately based ONLY on the provided context.
    - If the answer is in the context, provide a clear, concise answer.
    - After the answer, cite the sources used in a Markdown list.
    - If the answer cannot be found in the context, state that you cannot answer based on the provided information.
    - Do not make up information or use external knowledge.
    - Format your response using Markdown.
    """
    user_message = f"CONTEXT:\n{context_str}\n\n---\n\nQUESTION:\n{request.query}"

    async def stream_response():
        """Streams the response from OpenAI and includes source information at the end."""
        stream = await openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

        if sources:
            source_list = "\n".join([f"- {source}" for source in sources])
            yield f"\n\n---\n**Sources:**\n{source_list}"

    return StreamingResponse(stream_response(), media_type="text/plain; charset=utf-8")
```