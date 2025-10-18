from langgraph.graph import START, StateGraph
from typing import AsyncGenerator, List
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tools.movies import graph_rag_chain
from setup.init import NEO4J_URL, NEO4J_USERNAME, ANSWER_LLM
from utils.util import find_container_by_port
from urllib.parse import urlparse
import uvicorn, time, json, os

app = FastAPI(
    title="GraphRAG API", 
    version="1.2.0", 
    description="An API for querying a Neo4j knowledge graph using LLMs with retrieval-augmented generation (RAG)."
    )

# --- Middleware Configuration ---
# This allows your Streamlit frontend (running on a different port) to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class QueryRequest(BaseModel):
    question: str

@app.post("/")
def index():
    return {"message": "Welcome to the GraphRAG API. Use the /stream-ask endpoint to ask questions."}

@app.get("/api/v1/config")
def get_configuration():
    parsed_url = urlparse(NEO4J_URL)
    container_name = find_container_by_port(parsed_url.port)
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