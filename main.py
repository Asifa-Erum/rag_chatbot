import os
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient, AsyncQdrantClient
from openai import AsyncOpenAI
from fastapi.responses import StreamingResponse
import asyncio

# --- Configuration ---
def load_config():
    """Loads configuration from environment variables or config.json"""
    config = {}
    
    if os.getenv("OPENAI_API_KEY"):
        config["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    if os.getenv("QDRANT_URL"):
        config["QDRANT_URL"] = os.getenv("QDRANT_URL")
    if os.getenv("QDRANT_API_KEY"):
        config["QDRANT_API_KEY"] = os.getenv("QDRANT_API_KEY")
    if os.getenv("QDRANT_COLLECTION_NAME"):
        config["QDRANT_COLLECTION_NAME"] = os.getenv("QDRANT_COLLECTION_NAME")

    if not config or not all(k in config for k in ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]):
        try:
            script_dir = Path(__file__).parent
            config_path = script_dir / 'config.json'
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                for key in ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION_NAME"]:
                    if key not in config and key in file_config:
                        config[key] = file_config[key]
        except FileNotFoundError:
            pass
    
    config.setdefault("QDRANT_COLLECTION_NAME", "physical_ai_book_v2")
    return config

config = load_config()

# --- FastAPI App ---
app = FastAPI(
    title="Physical AI Textbook RAG Chatbot",
    description="A chatbot that answers questions based on the Physical AI & Humanoid Robotics textbook.",
    version="2.0.0",
)

# --- FIXED CORS (clean & correct) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://physical-ai-book-1.vercel.app",
        "https://rag-chatbot-eight-plum.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Clients ---
try:
    qdrant_client = AsyncQdrantClient(
        url=config["QDRANT_URL"],
        api_key=config["QDRANT_API_KEY"],
    )
    openai_client = AsyncOpenAI(api_key=config["OPENAI_API_KEY"])
    qdrant_collection_name = config["QDRANT_COLLECTION_NAME"]
except KeyError as e:
    print(f"FATAL: Missing required key in config.json: {e}")
    exit(1)
except Exception as e:
    print(f"Error initializing API clients: {e}")
    exit(1)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str
    selected_text: str | None = None

# --- RAG Logic ---
async def get_relevant_context(query: str, limit: int = 5) -> list[dict]:
    try:
        print(f"DEBUG: Searching Qdrant for query: '{query}'")
        query_embedding_response = await openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        )
        query_embedding = query_embedding_response.data[0].embedding

        search_results = qdrant_client.search(
            collection_name=qdrant_collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True,
            score_threshold=0.7
        )

        context = []
        for point in search_results:
            payload = point.payload
            context.append(
                {"text": payload.get("text", ""), "source": payload.get("source", "Unknown")}
            )
        return context

    except Exception as e:
        print(f"Error getting context: {e}")
        return []

# --- Endpoints ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "RAG Chatbot API is running"}

@app.post("/api/chat")
async def chat_handler(request: ChatRequest):

    if request.selected_text:
        context_str = request.selected_text
        sources = ["User-selected text"]
    else:
        relevant = await get_relevant_context(request.query)
        if relevant:
            context_str = "\n\n---\n\n".join([item["text"] for item in relevant])
            sources = sorted(list(set([item["source"] for item in relevant])))
        else:
            context_str = "No relevant context found."
            sources = []

    if context_str != "No relevant context found.":
        system_prompt = (
            "You are a helpful AI assistant for the Physical AI textbook. "
            "Answer concisely using ONLY the provided context. "
            "If context is insufficient, say: 'I cannot answer this from the provided information.'\n\n"
            f"Context:\n{context_str}\n\nSources: {', '.join(sources)}"
        )
    else:
        system_prompt = (
            "You are a helpful AI assistant. If context is missing, say 'I don't know.'"
        )

    async def stream():
        try:
            stream = await openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.query},
                ],
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

            if sources:
                yield f"\n\nSources: {', '.join(sources)}"

        except Exception as e:
            yield f"Error: {e}"

    return StreamingResponse(stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
