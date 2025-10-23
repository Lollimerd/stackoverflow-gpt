from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_neo4j import Neo4jGraph, Neo4jVector
from dotenv import load_dotenv
from typing import List
import os, json

# --- Configuration and Initialization (remains the same) ---
load_dotenv()
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASS")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

EMBEDDINGS = OllamaEmbeddings(model="bge-m3", base_url=OLLAMA_BASE_URL, num_ctx=8192)
graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, enhanced_schema=True)
# print(f"graph schema: {graph.schema}")

# This is a placeholder for LangChain's Document class
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
            f"Content: {doc.page_content}\n"
            f"--- METADATA ---\n"
            f"{metadata_str}"
        )
        formatted_blocks.append(block)

    # Join all the individual document blocks with a clear separator
    return "\n\n======================================================\n\n".join(formatted_blocks)

node_labels = ["User", "Tag", "Question", ""]

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
      
    )
print(f"Created vectorstore for {answerstore.index_name} index")

# test vectorstores, without retrieval query
print("Testing vectorstores...")
answers = answerstore.similarity_search("What is Flask?", k=5)
print(format_docs_with_metadata(answers), len(answers))

# init retrievers
# tag_retriever = tagstore.as_retriever(search_kwargs={'k': 20})
# user_retriever = userstore.as_retriever(search_kwargs={'k': 20})
# question_retriever = questionstore.as_retriever(search_kwargs={'k': 20})
# answer_retriever = answerstore.as_retriever(search_kwargs={'k': 20})

# ensemble_retriever = EnsembleRetriever(
#     retrievers=[tag_retriever, user_retriever, question_retriever, answer_retriever]
# )

# query = "What can you tell me about flask?"
# response = ensemble_retriever.invoke(query, k=5)
# print(format_docs_with_metadata(response), len(response))

# add pagerank
retrieval_query = """
// 1. Find the initial context entity from the Langchain retriever
WITH node AS s, score

// 2. Use a vector index to find semantically similar shows based on embeddings
// This is the core of the GraphRAG enhancement
CALL db.index.vector.queryNodes('Answer_index', 200, s.embeddings) YIELD node AS similar_answers, score AS answer_score
CALL db.index.vector.queryNodes('Question_index', 200, s.embeddings) YIELD node AS similar_questions, score AS question_score
CALL db.index.vector.queryNodes('Tag_index', 200, s.embeddings) YIELD node AS similar_tags, score AS tag_score
CALL db.index.vector.queryNodes('User_index', 200, s.embeddings) YIELD node AS similar_user, score AS user_score

// 3. From the similar shows, expand to get the full contextual neighborhood
WITH similar_answers, answer_score, s.pagerank AS original_pagerank
ORDER BY answer_score DESC

OPTIONAL MATCH (question:Question)<-[:ANSWERS]-(answer:Answer)
OPTIONAL MATCH (user:User)-[:ASKED]->(question:Question)
OPTIONAL MATCH (user:User)-[:PROVIDED]->(answer:Answer)
OPTIONAL MATCH (question:Question)-[:TAGGED]->(tag:Tag)

// 4. Create a combined score for ranking
// We boost the score based on the show's PageRank centrality
// This ensures that more "important" or popular shows rank higher
WITH similar_answers, similar_questions
    (answer_score * similar_answers.pagerank) AS combined_score,
    collect(DISTINCT question { .body, .title, .creation_date, .link }) AS questions,
    collect(DISTINCT answer { .score, .is_accepted, .creation_date, .body, .id }) AS answers,
    collect(DISTINCT user { .reputation, .id, .display_name }) AS user,
    collect(DISTINCT tag { .name }) AS tag,
ORDER BY combined_score DESC

// 5. Return the distinct text, metadata, relationships, and scores for the LLM
RETURN
    {answers: similar_answers { .score, .is_accepted, .creation_date, .body },
    questions: question { .body, .title, .creation_date, .link },
    user: user { .reputation, .id, .display_name },
    tag: tag.name
    } AS text,
    combined_score AS score,
    {   answers_id: similar_answers.id,
        answer_score: similar_answers.score,
        questions: questions,
        user: user.display_name,
        tag: tag.name,
        score: score
    } AS metadata,
"""