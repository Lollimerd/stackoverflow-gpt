# main.py
import asyncio, os, json, uvicorn, docker
from datetime import datetime
from typing import AsyncGenerator, List
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from urllib.parse import urlparse
from utils.util import format_docs_with_metadata, find_container_by_port
from setup.init import ANSWER_LLM, NEO4J_URL, NEO4J_USERNAME
from tools.custom_tool import graph_rag_chain

# ===========================================================================================================================================================
# FastAPI Backend Server 
# ===========================================================================================================================================================

# initialise fastapi 
app = FastAPI(title="GraphRAG API", version="1.2.0")

class QueryRequest(BaseModel):
    """configure question template for answer LLM"""
    question: str
    # chat_history: List[Dict[str, str]] # e.g., [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]

@app.get('/')
def index():
    return {"message": "Welcome to the GraphRAG API"}

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
    """This endpoint uses the agentic_router and re-implements the logic
    to parse <think> tags from the LLM's output stream, sending special
    tokens to the frontend to render the thinking process.
    """
    async def stream_generator() -> AsyncGenerator[str]:
        print(f"\n--- 💡 INCOMING QUESTION: '{request.question}' ---")

        async for chunk in graph_rag_chain.astream({"question": request.question}):
            # Extract content and reasoning
            content_chunk = chunk.content
            reasoning_chunk = chunk.additional_kwargs.get("reasoning_content", "")

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