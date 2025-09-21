# main.py
import asyncio, os, json, uvicorn, docker
from datetime import datetime
from typing import AsyncGenerator, List, Dict
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from pydantic import BaseModel
from urllib.parse import urlparse
from prompts import analyst_prompt
from utils.util import format_docs_with_metadata, find_container_by_port
from setup.init import (
    tagstore,
    answerstore,
    questionstore,
    userstore,
    EMBEDDINGS, ANSWER_LLM, graph,
    NEO4J_URL, NEO4J_USERNAME,
)

# ===========================================================================================================================================================
# Setting Up Retrievers from vectorstores for EnsembleRetriever 
# ===========================================================================================================================================================

# setting up retrievers from vectorstores with custom tailormade finetuning
def retrieve_context(question: str) -> List[Document]:
    """Retrieve context from the ensemble retriever based on the user's question."""
    # init retrievers
    tag_retriever = tagstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'k': 10, 
                    'params': {'embedding': EMBEDDINGS.embed_query(question)},
                    'fetch_k': 100,
                    'score_threshold': 0.70,
                    'lambda_mult': 0.5,
                    })

    user_retriever = userstore.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={'k': 10, 
                'params': {'embedding': EMBEDDINGS.embed_query(question)},
                'fetch_k': 100,
                'score_threshold': 0.70,
                'lambda_mult': 0.5,
                })

    question_retriever = questionstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'k': 10, 
                    'params': {'embedding': EMBEDDINGS.embed_query(question)},
                    'fetch_k': 100,
                    'score_threshold': 0.70,
                    'lambda_mult': 0.5,
                    })

    answer_retriever = answerstore.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={'k': 10, 
                    'params': {'embedding': EMBEDDINGS.embed_query(question)},
                    'fetch_k': 100,
                    'score_threshold': 0.70,
                    'lambda_mult': 0.5,
                    })

    ensemble_retriever = EnsembleRetriever(
        retrievers=[tag_retriever, user_retriever, question_retriever, answer_retriever]
    )

    print("---RETRIEVING CONTEXT---")
    # return vectorstore.similarity_search(question, k=6)
    return ensemble_retriever.invoke(question, k=3)

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

# and the provided context inside <think></think> tags
@app.post("/stream-ask")
async def stream_ask_question(request: QueryRequest) -> StreamingResponse:
    async def stream_generator() -> AsyncGenerator[str, None]:
        # --- Prepare context and chain (remains the same) ---
        context_docs = await asyncio.to_thread(retrieve_context, request.question)
        context_str = format_docs_with_metadata(context_docs)
        print(context_str)
        print(f"documents retrieved: {len(context_docs)}")
        
        print("\nSetting up Pipeline...")
        chain = analyst_prompt | ANSWER_LLM
        print("\npreparing streaming answers")

        # Special tags to signal thought blocks to the frontend
        THINK_START_TAG = "<|THINK_START|>"
        THINK_END_TAG = "<|THINK_END|>"

        # --- Streaming logic ---
        buffer = ""
        # Get the raw stream of content from the LLM
        async for chunk in chain.astream({"question": request.question, "context": context_str}):
            buffer += chunk.content

            # Check for <think> tags and replace them with our special tags
            if "<think>" in buffer:
                # Isolate the text before the tag and yield it
                pre_think_text, think_block = buffer.split("<think>", 1)
                yield pre_think_text
                # Yield the start tag and update the buffer
                yield THINK_START_TAG
                buffer = think_block
                print("model thinking")

            if "</think>" in buffer:
                # Isolate the thought content
                thought_content, post_think_text = buffer.split("</think>", 1)
                yield thought_content
                # Yield the end tag and update the buffer
                yield THINK_END_TAG
                buffer = post_think_text
                print("outputing answers")

            # Yield any text that doesn't contain tags
            if THINK_START_TAG not in buffer and THINK_END_TAG not in buffer:
                yield buffer
                buffer = ""

        # Yield any final text remaining in the buffer
        if buffer:
            yield buffer
    
    print("done")
    # The media type is now plain text
    return StreamingResponse(stream_generator(), media_type="application/json")

# uvicorn main:app --reload
if __name__ == "__main__":
    # Run the app with Uvicorn, specifying host and port here
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")