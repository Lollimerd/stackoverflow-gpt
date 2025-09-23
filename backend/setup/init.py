"""Setting up ollama models, vectorstores and Neo4j Configs"""

import os, json, time
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain, Neo4jVector
from langchain.chains.summarize import load_summarize_chain
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from typing import Dict, Optional
from langchain_core.documents import Document

# ===========================================================================================================================================================
# Step 1: Load Configuration: Docker, Neo4j, Ollama, Langchain
# ===========================================================================================================================================================

load_dotenv()
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

# qwen3:8b works for now with limited context of 40k, qwen3:30b works with 256k max
ANSWER_LLM = ChatOllama(
    model="qwen3:1.7b", # Ensure your model produces <think> tags
    base_url=OLLAMA_BASE_URL, 
    num_ctx=40968, # 32k
    num_predict=-2, # fill context
    tfs_z=2.0, # reduce impact of less probable tokens from output
    repeat_penalty=1.5, # higher, penalise repetitions
    repeat_last_n=-1, # look back within context to penalise penalty
    top_p=0.95, # more diverse text
    top_k=100, # give more diverse answers
    mirostat=2.0, # enable mirostat 2.0 sampling for controlling perplexity
    mirostat_tau=8.0, # output diversity
    mirostat_eta=0.05, # learning rate, responsiveness
) 

EMBEDDINGS = OllamaEmbeddings(
    model="bge-m3", 
    base_url=OLLAMA_BASE_URL, 
    show_progress=True, 
    tfs_z=2.0, # reduce impact of less probable tokens from output
    mirostat=2.0, # enable mirostat 2.0 sampling for controlling perplexity
    mirostat_tau=1.0, # output diversity consistent
    mirostat_eta=0.2 # faster learning rate 
)

graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    enhanced_schema=True,
    refresh_schema=True
)
print(f"\nschema: {graph.schema}\n")

# define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# test embedding generation
# sample_embedding = EMBEDDINGS.embed_query("Hello world")
# print(f"\nSample embedding dimension: {len(sample_embedding)}")

# ===========================================================================================================================================================
# Crafting custom cypher retrieval queries
# ===========================================================================================================================================================

retrieval_query = """
// 1. Perform a vector similarity search on the 'Question_index'
CALL db.index.vector.queryNodes('Question_index', 50, $embedding) YIELD node AS question, score AS similarityScore

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

# ===========================================================================================================================================================
# Creation of vector index, vectorstores and fulltext for hybrid vector search
# ===========================================================================================================================================================

def create_vector_stores(graph, EMBEDDINGS, retrieval_query) -> Dict[str, Neo4jVector]:
    """
    Creates Neo4jVector stores from an existing graph using a data-driven approach.

    Args:
        graph: The Neo4j graph instance.
        EMBEDDINGS: The embedding model.
        retrieval_query: The Cypher query for retrieval.

    Returns:
        A dictionary of Neo4jVector store instances, keyed by their node label.
    """
    
    # Define a list of configurations for each vector store
    store_configs = [
        {
            "node_label": "Tag",
            "text_node_properties": ["name"],
        },
        {
            "node_label": "User",
            "text_node_properties": ["reputation", "display_name"],
        },
        {
            "node_label": "Question",
            "text_node_properties": ["score", "link", "favourite_count", "id", "creation_date", "body", "title"],
        },
        {
            "node_label": "Answer",
            "text_node_properties": ["score", "is_accepted", "id", "body", "creation_date"],
        },
    ]

    vectorstores = {}
    
    # Loop through the configurations and create the vector stores
    for config in store_configs:
        label = config["node_label"]
        index_name = f"{label}_index"
        keyword_index_name = f"{label}_keyword_index"
        
        vectorstores[label.lower() + "store"] = Neo4jVector.from_existing_graph(
            graph=graph,
            node_label=label,
            embedding=EMBEDDINGS,
            embedding_node_property="embedding",
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type="hybrid",
            text_node_properties=config["text_node_properties"],
            retrieval_query=retrieval_query,
        )
        print(f"Created vectorstore for {index_name} index")
        
    return vectorstores

# Create vector stores
stores = create_vector_stores(graph, EMBEDDINGS, retrieval_query)
tagstore = stores['tagstore']
userstore = stores['userstore']
questionstore = stores['questionstore']
answerstore = stores['answerstore']