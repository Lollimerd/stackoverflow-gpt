from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph, Neo4jVector
from matplotlib.style import context
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from dotenv import load_dotenv
import asyncio, os, json
from typing import AsyncGenerator, List
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# --- Configuration and Initialization (remains the same) ---
load_dotenv()
NEO4J_URL = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

# qwen3:8b works for now with limited context of 40k out of 128k
ANSWER_LLM = ChatOllama(model="qwen3:1.7b", base_url=OLLAMA_BASE_URL, num_ctx=40960)
EMBEDDINGS = OllamaEmbeddings(model="bge-m3", base_url=OLLAMA_BASE_URL, num_ctx=8192)

graph = Neo4jGraph(
    url=NEO4J_URL, 
    username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD, 
    enhanced_schema=True
    )

# node and metadata details
node_labels = ["Person", "Genre", "Country"]
show_metadata = ["duration", "release_year", "rating", "type", "title", "show_id", "description"]

retrieval_query = """
// Alias the `node` provided by Langchain to `s` for clarity, and pass along the `score`.
WITH node AS s, score

// Use OPTIONAL MATCH in case a show doesn't have a director or cast
OPTIONAL MATCH (s:Show)<-[:DIRECTED]-(p:Person)
OPTIONAL MATCH (s:Show)<-[:ACTED_IN]-(a:Person)
OPTIONAL MATCH (s:Show)-[:IN_GENRE]->(g:Genre)
OPTIONAL MATCH (s:Show)-[:PRODUCED_IN]->(country:Country)

// Use coalesce to handle sentences not linked to an Exercise.
WITH coalesce(s, p, a, g) AS context_entity, score
ORDER BY score DESC

// Expand from the context_entity to retrieve the full contextual neighborhood.
CALL {
    WITH context_entity
    MATCH (related_s:Show)

    WHERE (context_entity:Person AND EXISTS((related_s)-[]-(context_entity)))
    OR (context_entity:Country AND EXISTS((related_s)-[]-(context_entity)))
    OR (context_entity:Genre AND EXISTS((related_s)-[]-(context_entity)))

    // Condition x: Or, if the entity was just a Sentence, return it.
    OR related_s = context_entity
    RETURN related_s
}

// For each show, collect its relationships
WITH DISTINCT related_s, score
OPTIONAL MATCH (director:Person)-[:DIRECTED]->(related_s)
OPTIONAL MATCH (actor:Person)-[:ACTED_IN]->(related_s)
OPTIONAL MATCH (related_s)-[:IN_GENRE]->(genre:Genre)
OPTIONAL MATCH (related_s)-[:PRODUCED_IN]->(country:Country)

WITH related_s, score,
     collect(DISTINCT director { .name }) AS directors,
     collect(DISTINCT actor { .name }) AS cast,
     collect(DISTINCT genre { .name }) AS genres,
     collect(DISTINCT country { .name }) AS countries

// Return the distinct text and metadata for the LLM.
RETURN DISTINCT related_s {.duration, .release_year, .rating, .description, .title} AS text,
    related_s {.show_id, score: score, directed_by: directors,
        acted_in_by: cast,
        has_genre: genres,
        produced_in: countries} AS metadata,
    score,
    {
        directed_by: directors,
        acted_in_by: cast,
        has_genre: genres,
        produced_in: countries
    } as relationships
"""

# create vectorstores from vector index
show_vectorstore = Neo4jVector.from_existing_index(
    graph=graph,
    embedding=EMBEDDINGS,
    retrieval_query=retrieval_query,
    embedding_node_property="embeddings",
    index_name="show_index",
    keyword_index_name="Show_keyword_index",
    search_type="hybrid",
)

country_vectorstore = Neo4jVector.from_existing_index(
    graph=graph,
    embedding=EMBEDDINGS,
    retrieval_query=retrieval_query,
    embedding_node_property="embeddings",
    index_name="Country_index",
    keyword_index_name="Country_keyword_index",
    search_type="hybrid"
)

genre_vectorstore = Neo4jVector.from_existing_index(
    graph=graph,
    embedding=EMBEDDINGS,
    retrieval_query=retrieval_query,
    embedding_node_property="embeddings",
    index_name="Genre_index",
    keyword_index_name="Genre_keyword_index",
    search_type="hybrid"
)
person_vectorstore = Neo4jVector.from_existing_index(
    graph=graph,
    embedding=EMBEDDINGS,
    retrieval_query=retrieval_query,
    embedding_node_property="embeddings",
    index_name="Person_index",
    keyword_index_name="Person_keyword_index",
    search_type="hybrid"
)

# configure retrievers from vectorstores
show_retriever = show_vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k': 20, 'score_threshold': 1.0, 'params': {'node_label': show_vectorstore.node_label}})
country_retriever = country_vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k': 20, 'score_threshold': 1.0})
genre_retriever = genre_vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k': 20, 'score_threshold': 1.0})
person_retriever = person_vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k': 20, 'score_threshold': 1.0})

# init ensemble retriever
combined_retriever = EnsembleRetriever(
    retrievers=[show_retriever, country_retriever, genre_retriever, person_retriever],
    # weight=[],
)

app = FastAPI(title="GraphRAG API", version="1.2.0")

class QueryRequest(BaseModel):
    question: str

def retrieve_context(question: str) -> List[Document]:
    print("---RETRIEVING CONTEXT---")
    # return vectorstore.similarity_search(question, k=6)
    return combined_retriever.invoke(question, k=6)

class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

def format_docs_with_metadata(docs: List[Document]) -> str:
    """
    Formats a list of Documents into a single string, where each
    document's page_content is followed by its corresponding metadata.
    """
    # Create a list of formatted strings, one for each document
    formatted_blocks = []
    for doc in docs:
        # Format the metadata as a pretty JSON string
        metadata_str = json.dumps(doc.metadata, indent=2)

        # Create a combined block for the document's content and its metadata
        block = (
            f"Page Content: {doc.page_content}\n"
            f"--------- METADATA ---------\n"
            f"{metadata_str}"
        )
        formatted_blocks.append(block)

    # Join all the individual document blocks with a clear separator
    return "\n\n======================================================\n\n".join(formatted_blocks)

# and the provided context inside <think></think> tags
@app.post("/stream-ask")
async def stream_ask_question(request: QueryRequest) -> StreamingResponse:
    async def stream_generator() -> AsyncGenerator[str, None]:
        # Special tags to signal thought blocks to the frontend
        THINK_START_TAG = "<|THINK_START|>"
        THINK_END_TAG = "<|THINK_END|>"

        # --- Prepare context and chain (remains the same) ---
        context_docs = await asyncio.to_thread(retrieve_context, request.question)
        context_str = format_docs_with_metadata(context_docs)
        print(context_str)
        print(f"Documents retrieved: {len(context_docs)}")
        template = """
        <｜begin of sentence｜>
        You are an expert Analyst AI. First, think step-by-step about the user's question and the provided context.
        Your mission is to analyze the structured data retrieved from a Neo4j knowledge graph. 
        Your primary function is to go beyond simple summarization. You must infer connections and extrapolate potential outcomes
        If there is no or not enough context given, state so clearly.
        When presenting tabular data, please format it as a Github-flavored Markdown table.
        When the user's question is best answered with a diagram (like a flowchart, sequence, or hierarchy), generate the diagram using Mermaid syntax. 
        After your thought process, provide the final, concise answer to the user based on your analysis.

        **Context from Knowledge Graph:**
        {context}

        **User's Question:**
        {question}
        <｜end of sentence｜>
        <｜begin of sentence｜>
        """
        prompt = PromptTemplate.from_template(template)
        chain = prompt | ANSWER_LLM

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