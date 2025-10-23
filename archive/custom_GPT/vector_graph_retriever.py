from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph, Neo4jVector
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from dotenv import load_dotenv
import asyncio, os

# --- Configuration and Initialization (remains the same) ---
load_dotenv()
NEO4J_URL = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

# qwen3:8b works for now with limited context of 40k out of 128k
ANSWER_LLM = ChatOllama(model="qwen3:8b", base_url=OLLAMA_BASE_URL, num_ctx=40960)
EMBEDDINGS = OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_BASE_URL)

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

// Return the distinct text and metadata for the LLM.
RETURN DISTINCT related_s.description AS text,
       related_s { .duration, .release_year, .rating, .show_id} AS metadata, 
       score limit 150
"""

# create vectorstores from vector index
show_vectorstore = Neo4jVector.from_existing_index(
    graph=graph,
    embedding=EMBEDDINGS,
    retrieval_query=retrieval_query,
    embedding_node_property="embeddings",
    index_name="show_index",
    keyword_index_name="Show_keyword_index",
)

country_vectorstore = Neo4jVector.from_existing_index(
    graph=graph,
    embedding=EMBEDDINGS,
    retrieval_query=retrieval_query,
    embedding_node_property="embeddings",
    index_name="Country_index",
    keyword_index_name="Country_keyword_index",
)

genre_vectorstore = Neo4jVector.from_existing_index(
    graph=graph,
    embedding=EMBEDDINGS,
    retrieval_query=retrieval_query,
    embedding_node_property="embeddings",
    index_name="Genre_index",
    keyword_index_name="Country_keyword_index",
)
person_vectorstore = Neo4jVector.from_existing_index(
    graph=graph,
    embedding=EMBEDDINGS,
    retrieval_query=retrieval_query,
    embedding_node_property="embeddings",
    index_name="Person_index",
    keyword_index_name="Country_keyword_index",
)

# configure retrievers from vectorstores
show_retriever = show_vectorstore.as_retriever(search_kwargs={'k': 6})
country_retriever = country_vectorstore.as_retriever(search_kwargs={'k': 6})
genre_retriever = genre_vectorstore.as_retriever(search_kwargs={'k': 6})
person_retriever = person_vectorstore.as_retriever(search_kwargs={'k': 6})

# init ensemble retriever
combined_retriever = EnsembleRetriever(
    retrievers=[show_retriever, country_retriever, genre_retriever, person_retriever],
    # weight=[],
)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Create a prompt
template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}

Answer:"""

# prompt template
prompt = PromptTemplate.from_template(template)

# Retrieve context 
def retrieve(state: State):
    # Use the vector to find relevant documents
    context = combined_retriever.invoke(state["question"], k=3,)
    return {"context": context}

# Generate the answer based on the question and context
def generate(state: State):
    messages = prompt.invoke({"question": state["question"], "context": state["context"]})
    response = ANSWER_LLM.invoke(messages)
    return {"answer": response.content}

# Define application steps
workflow = StateGraph(State).add_sequence([retrieve, generate])
workflow.add_edge(START, "retrieve")
app = workflow.compile()

# Run the application
question = "What can you tell me about the movie LOVE"
response = app.invoke({"question": question})
print("\nAnswer:", response["answer"])
# print("Context:", response["context"])