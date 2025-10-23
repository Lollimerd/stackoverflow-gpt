from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph, Neo4jVector
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from dotenv import load_dotenv
import asyncio, os, json, uvicorn, tqdm, docker
from typing import AsyncGenerator, List
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.retrievers import EnsembleRetriever
from utils import format_docs_with_metadata
from urllib.parse import urlparse

# --- Configuration and Initialization (remains the same) ---
load_dotenv()
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASS")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

# qwen3:8b works for now with limited context of 40k out of 128k
ANSWER_LLM = ChatOllama(model="qwen3:1.7b", base_url=OLLAMA_BASE_URL, num_ctx=40960, num_predict=-2)
EMBEDDINGS = OllamaEmbeddings(model="bge-m3", base_url=OLLAMA_BASE_URL, num_ctx=8192, show_progress=True)

graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, enhanced_schema=True)
# print(graph.schema)

retrieval_query = """
// 1. Perform a vector similarity search on the 'Question_index'
CALL db.index.vector.queryNodes('Question_index', 30, $embedding) YIELD node AS question, score AS similarityScore

// 2. Fetch the user who asked the question
OPTIONAL MATCH (asker:User)-[:ASKED]->(question)

// 3. Collect all tags associated with the question
OPTIONAL MATCH (question)-[:TAGGED]->(tag:Tag)
WITH question, similarityScore, asker, COLLECT(DISTINCT tag.name) AS tags

// 4. Collect all answers, and the users who provided them
OPTIONAL MATCH (answer:Answer)-[:ANSWERS]->(question)
OPTIONAL MATCH (provider:User)-[:PROVIDED]->(answer)
WITH question, similarityScore, asker, tags, COLLECT(DISTINCT {
    id: answer.id,
    body: answer.body,
    score: answer.score,
    is_accepted: answer.is_accepted,
    creation_date: toString(answer.creation_date),
    provided_by: {
        id: provider.id,
        display_name: provider.display_name,
        reputation: provider.reputation
    }
}) AS answerDetails

// 5. Format and return the final results
RETURN
    // The primary text content for retrieval
    'Title:\n ' + question.title + '\nBody: ' + question.body AS text,
    
    // The structured metadata containing all graph context
    {
        question_details: {
            id: question.id,
            title: question.title,
            link: question.link,
            score: question.score,
            favorite_count: question.favorite_count,
            creation_date: toString(question.creation_date)
        },
        asked_by: {
            id: asker.id,
            display_name: asker.display_name,
            reputation: asker.reputation
        },
        tags: tags,
        answers: answerDetails,
        simscore: similarityScore
    } AS metadata,
    
    // The similarity score from the initial vector search
    similarityScore as score
ORDER BY score DESC
"""

# creating vector index for vectorstores
tagstore = Neo4jVector.from_existing_graph(
        graph=graph,
        node_label="Tag",
        embedding=EMBEDDINGS,
        embedding_node_property="embeddings",
        index_name="Tag_index",
        keyword_index_name="Tag_keyword_index",
        search_type="hybrid",
        text_node_properties=["name"],
        retrieval_query=retrieval_query,
    )
print(f"Created vectorstore for {tagstore.index_name} index")

userstore = Neo4jVector.from_existing_graph(
        graph=graph,
        node_label="User",
        embedding=EMBEDDINGS,
        embedding_node_property="embeddings",
        index_name="User_index",
        keyword_index_name="User_keyword_index",
        search_type="hybrid",
        text_node_properties=["reputation", "display_name"],
        retrieval_query=retrieval_query,
        
    )
print(f"Created vectorstore for {userstore.index_name} index")

questionstore = Neo4jVector.from_existing_graph(
        graph=graph,
        node_label="Question",
        embedding=EMBEDDINGS,
        embedding_node_property="embeddings",
        index_name="Question_index",
        keyword_index_name="Question_keyword_index",
        search_type="hybrid",
        text_node_properties=["score", "link", "favourite_count", "id", "creation_date", "body", "title"],
        retrieval_query=retrieval_query,
    )
print(f"Created vectorstore for {questionstore.index_name} index")

answerstore = Neo4jVector.from_existing_graph(
        graph=graph,
        node_label="Answer",
        embedding=EMBEDDINGS,
        embedding_node_property="embeddings",
        index_name="Answer_index",
        keyword_index_name="Answer_keyword_index",
        search_type="hybrid",
        text_node_properties=["score", "is_accepted", "id", "body", "creation_date"],
        retrieval_query=retrieval_query,
    )
print(f"Created vectorstore for {answerstore.index_name} index")

app = FastAPI(title="GraphRAG API", version="1.2.0")

class QueryRequest(BaseModel):
    question: str

def retrieve_context(question: str) -> List[Document]:
    """Retrieve context from the ensemble retriever based on the user's question."""
    # init retrievers
    tag_retriever = tagstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'k': 6, 
                    'params': {'embedding': EMBEDDINGS.embed_query(question)},
                    'fetch_k': 50,
                    'score_threshold': 0.75,
                    'lambda_mult': 0.4,
                    })

    user_retriever = userstore.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={'k': 6, 
                'params': {'embedding': EMBEDDINGS.embed_query(question)},
                'fetch_k': 50,
                'score_threshold': 0.75,
                'lambda_mult': 0.4,
                })

    question_retriever = questionstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'k': 6, 
                    'params': {'embedding': EMBEDDINGS.embed_query(question)},
                    'fetch_k': 50,
                    'score_threshold': 0.75,
                    'lambda_mult': 0.4,
                    })

    answer_retriever = answerstore.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={'k': 6, 
                    'params': {'embedding': EMBEDDINGS.embed_query(question)},
                    'fetch_k': 50,
                    'score_threshold': 0.75,
                    'lambda_mult': 0.4,
                    })

    ensemble_retriever = EnsembleRetriever(
        retrievers=[tag_retriever, user_retriever, question_retriever, answer_retriever]
    )

    print("---RETRIEVING CONTEXT---")
    # return vectorstore.similarity_search(question, k=6)
    return ensemble_retriever.invoke(question, k=3)

@app.get('/')
def get_root():
    return {"message": "Welcome to the GraphRAG API"}

# --- Add this config endpoint ---
@app.get("/api/v1/config")
def get_configuration():
    """
    Provides frontend with configuration details for display.
    """
    return {
        "ollama_model": ANSWER_LLM.model,
        "neo4j_url": NEO4J_URL,
        "neo4j_user": NEO4J_USERNAME,
    }

# --- Dynamic Container Discovery ---
def find_container_by_port(port: int) -> str:
    """
    Inspects running Docker containers to find which one is using the specified port.
    """
    if not port:
        return "Invalid port"
    try:
        # Connect to the Docker daemon
        client = docker.from_env()
        containers = client.containers.list()

        for container in containers:
            # The .ports attribute is a dictionary like: {'7687/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '7687'}]}
            port_mappings = container.ports
            for container_port, host_mappings in port_mappings.items():
                if host_mappings:
                    for mapping in host_mappings:
                        if mapping.get("HostPort") == str(port):
                            return container.name # Found it!

        return "No matching container found"

    except docker.errors.DockerException:
        return "Docker daemon not running or not accessible"
    except Exception as e:
        return f"An error occurred: {e}"

# New, smarter endpoint to provide the container name
@app.get("/neo4j-container-name")
async def get_neo4j_container_name():
    """
    Parses the NEO4J_URL, extracts the port, and finds the container using it.
    """
    parsed_url = urlparse(NEO4J_URL)
    neo4j_port = parsed_url.port
    container_name = find_container_by_port(neo4j_port)
    return {"container_name": container_name}

# and the provided context inside <think></think> tags
@app.post("/stream-ask")
async def stream_ask_question(request: QueryRequest) -> StreamingResponse:
    """streams the response from the LLM, including thought blocks"""
    async def stream_generator() -> AsyncGenerator[str, None]:
        """Generator to stream the response from the LLM with thought blocks"""
        # Special tags to signal thought blocks to the frontend
        THINK_START_TAG = "<|THINK_START|>"
        THINK_END_TAG = "<|THINK_END|>"

        # --- Prepare context and chain (remains the same) ---
        context_docs = await asyncio.to_thread(retrieve_context, request.question)
        context_str = format_docs_with_metadata(context_docs)
        print(f"Context retrieved: {context_str}")
        print("documents retrieved:", len(context_docs))
        template = """
        <|im_start|>system
        You are an tech expert AI. First, think step-by-step about the user's question and the provided context.
        Your mission is to use the context to help answer the user's question, from stackoverflow context
        Your primary function is to go beyond simple summarization. You must infer connections and provide reasoning.
        If there is no or not enough context given, state so clearly and compensate with your own knowledge
        If the user question is totally not related to the context, just answer back normally
        When presenting tabular data, please format it as a Github-flavored Markdown table.
        When the user's question is best answered with a diagram (like a flowchart, sequence, or hierarchy), generate the diagram using Mermaid syntax. 
        After your thought process, provide the final answer to the user based on your analysis
        <|im_end|>
        
        <|im_start|>user
        Hi
        <|im_end|>
        <|im_start|>assistant
        Hello! How can I assist you today?
        <|im_end|>

        <|im_start|>context
        {context}
        <|im_end|>

        <|im_start|>user
        {question}
        <|im_end|>

        <|im_start|>assistant
        """
        print("prompting llm...")
        prompt = PromptTemplate.from_template(template)
        chain = prompt | ANSWER_LLM
        print("invoke pipeline")

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

            # Yield any text that doesn't contain tags
            if THINK_START_TAG not in buffer and THINK_END_TAG not in buffer:
                yield buffer
                buffer = ""

        # Yield any final text remaining in the buffer
        if buffer:
            yield buffer

    # The media type is now plain text
    return StreamingResponse(stream_generator(), media_type="application/json")

# This command starts the API server
# uvicorn main:app --reload

uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")