from langchain_community.embeddings import OllamaEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from dotenv import load_dotenv
import os

# --- Configuration and Initialization (remains the same) ---
load_dotenv()
NEO4J_URL = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

EMBEDDINGS = OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_BASE_URL)
graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, enhanced_schema=True)
print(f"graph schema: {graph.schema}")

node_labels = ["Person", "Genre", "Country"]
show_metadata = ["duration", "release_year", "rating", "type", "title", "show_id", "description"]

# creating vector index for vectorstores
# for i in node_labels:
#     print(f"Creating vectorstore for {i}...")
#     vectorstore = Neo4jVector.from_existing_graph(
#         graph=graph,
#         node_label=i,
#         embedding=EMBEDDINGS,
#         embedding_node_property="embeddings",
#         index_name=f"{i}_index",
#         keyword_index_name=f"{i}_keyword_index",
#         search_type="hybrid",
#         text_node_properties=["name"],
#     )
#     print(f"Created vectorstore for {i} with index {vectorstore.index_name}")

# mainstore = Neo4jVector.from_existing_graph(
#     graph=graph,
#     node_label="Show",
#     embedding=EMBEDDINGS,
#     embedding_node_property="embeddings",
#     index_name="Show_index",
#     keyword_index_name="Show_keyword_index",
#     search_type="hybrid",
#     text_node_properties=show_metadata,
# )
# print(f"Created main vectorstore with index {mainstore.index_name}")