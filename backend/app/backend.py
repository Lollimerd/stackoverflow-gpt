# main.py
import asyncio, os, json, uvicorn, docker
from datetime import datetime
from typing import AsyncGenerator, List, Dict
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from urllib.parse import urlparse
from dotenv import load_dotenv
from utils.util import format_docs_with_metadata, find_container_by_port
from setup.init import ANSWER_LLM, NEO4J_URL, NEO4J_USERNAME
from app.graph import graph_rag_chain
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

# Load environment variables
load_dotenv()

LANGFUSE_PUBLIC_KEY = os.getenv('LANGFUSE_PUBLIC_KEY')

# Initialize Langfuse client                                                                                                                                                                                                                          │
langfuse_client = get_client(public_key=LANGFUSE_PUBLIC_KEY)
langfuse_handler = CallbackHandler(update_trace=True)

# ===========================================================================================================================================================
# FastAPI Backend Server 
# ===========================================================================================================================================================

# initialise fastapi 
app = FastAPI(title="GraphRAG API", version="1.2.0")

class QueryRequest(BaseModel):
    """configure question template for answer LLM"""
    question: str
    chat_history: List[Dict[str, str]] = []
    session_id: str = None  # Optional session ID for tracking

@app.get('/')
def index():
    return {"message": "Welcome to the GraphRAG API"}

@app.get("/health")
def health():
    """Check if the Langfuse connection is healthy."""
    if langfuse_client is None:
        return {"status": "error", "message": "Langfuse handler not initialized"}
    
    try:
        # Attempt to use a feature that requires a connection
        langfuse_client.auth_check()
        return {"status": "ok", "message": "Langfuse connection is healthy"}
    except Exception as e:
        return {"status": "error", "message": f"Langfuse connection failed: {e}"}

# --- Add config endpoint ---
@app.get("/api/v1/config")
def get_configuration():
    """Provides frontend with configuration details for display."""
    parsed_url = urlparse(NEO4J_URL)
    neo4j_port = parsed_url.port
    container_name = find_container_by_port(neo4j_port)
    return {
        "ollama_model": ANSWER_LLM.model,
        "neo4j_url": NEO4J_URL,
        "container_name": container_name,
        "neo4j_user": NEO4J_USERNAME,
    }

# --- ✨ REFACTORED: Streaming Endpoint with Thinking Handler ---
@app.post("/stream-ask")
async def stream_ask_question(request: QueryRequest) -> StreamingResponse:
    """This endpoint now includes chat history for context-aware responses."""
    async def stream_generator() -> AsyncGenerator[str]:
        print(f"\n--- 💡 INCOMING QUESTION: '{request.question}' ---")
        if len(request.chat_history) == 0:
            print(f"--- 📚 CHAT HISTORY: {len(request.chat_history)} messages ---")
        else: # excluding current message
            print(f"--- 📚 CHAT HISTORY: {len(request.chat_history)-1} messages ---")

        # Configure Langfuse with session tracking and metadata
        langfuse_config = {
            "callbacks": [langfuse_handler],
            "metadata": {
                "session_id": request.session_id,
                "question_length": len(request.question),
                "history_length": len(request.chat_history),
            },
            "tags": ["graphrag", "stackoverflow"]
        }
        
        if request.session_id:
            langfuse_config["run_name"] = f"query_{request.session_id}"

        # Pass chat history to the chain
        async for chunk in graph_rag_chain.astream(
            {"question": request.question, "chat_history": request.chat_history},
            config=langfuse_config,
        ):
            if "answer" in chunk:
                # Extract content and reasoning
                content_chunk = chunk["answer"].content
                reasoning_chunk = chunk["answer"].additional_kwargs.get("reasoning_content", "")

                # Create a dictionary for this stream chunk
                event_data = {
                    "content": content_chunk,
                    "reasoning_content": reasoning_chunk,
                }

                # Format as an SSE data payload
                yield f"data: {json.dumps(event_data)}\n\n"
            
    # The media type is now plain text to stream the raw content and tags.
    return StreamingResponse(stream_generator(), media_type="text/event-stream")

# uvicorn main:app --reload
if __name__ == "__main__":
    # Run the app with Uvicorn, specifying host and port here
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")