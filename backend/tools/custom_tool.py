from setup.init import graph, EMBEDDINGS, create_vector_stores, ANSWER_LLM, RERANKER_MODEL
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from typing import List
from langchain_core.documents import Document
from prompts.st_overflow import analyst_prompt
from langchain_core.tools import Tool
from langchain_core.runnables import RunnablePassthrough
from utils.util import format_docs_with_metadata

# ===========================================================================================================================================================
# Crafting custom cypher retrieval queries
# ===========================================================================================================================================================
retrieval_query = """
// Base query starting with vector search results
MATCH (question:Question)
WHERE question.communityId[0] IN node.communityId
WITH question, score

// Core question data
WITH question, score, {
    id: question.id,
    title: question.title,
    body: question.body,
    link: question.link,
    score: question.score,
    favorite_count: question.favorite_count,
    creation_date: toString(question.creation_date),
    communityId: question.communityId
} AS questionDetails

// Fetch asker details
OPTIONAL MATCH (asker:User)-[:ASKED]->(question)
WITH question, score, questionDetails, {
    id: asker.id,
    display_name: asker.display_name,
    reputation: asker.reputation
} AS askerDetails

// Collect tags
OPTIONAL MATCH (question)-[:TAGGED]->(tag:Tag)
WITH question, score, questionDetails, askerDetails, 
     COLLECT(DISTINCT tag.name) AS tags

// Collect answers with their providers
OPTIONAL MATCH (answer:Answer)-[:ANSWERS]->(question)
OPTIONAL MATCH (provider:User)-[:PROVIDED]->(answer)
WITH question, score, questionDetails, askerDetails, tags,
     COLLECT(DISTINCT {
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
     }) AS answers

// Return formatted results
RETURN 
    'Title: ' + question.title + '\nBody: ' + question.body AS text,
    {
        question_details: questionDetails,
        asked_by: askerDetails,
        tags: tags,
        answers: {
            answers: answers,
            communityId: question.communityId
        },
        simscore: score
    } AS metadata,
    score
ORDER BY score DESC
LIMIT 5000
"""

# Create vector stores
stores = create_vector_stores(graph, EMBEDDINGS, retrieval_query)
tagstore = stores['tagstore']
userstore = stores['userstore']
questionstore = stores['questionstore']
answerstore = stores['answerstore']

# set up reranker model
compressor = CrossEncoderReranker(
        model=RERANKER_MODEL,
        top_n=10  # This will return the top n most relevant documents.
    )

# ===========================================================================================================================================================
# Setting Up Retrievers from vectorstores for EnsembleRetriever 
# ===========================================================================================================================================================

# setting up retrievers from vectorstores with custom tailormade finetuning
def retrieve_context(question: str) -> List[Document]:
    """
    Retrieve context from the ensemble retriever based on the user's question.

    Returns:
        List[Document]: A list of LangChain Document objects, where each document has:
        - page_content (str): The main text content (e.g., question title and body).
        - metadata (dict): Structured metadata including question details, user info, tags, answers, and similarity score.
    """
    
    # Define the common search arguments once
    common_search_kwargs = {
        'k': 15,
        'params': {'embedding': EMBEDDINGS.embed_query(question)},
        'fetch_k': 100,
        'score_threshold': 0.85,
        'lambda_mult': 0.5,
    }

    # Use a list of vectorstores
    vectorstores = [tagstore, userstore, questionstore, answerstore]

    # Create the retrievers using a list comprehension
    retrievers = [
        store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=common_search_kwargs
        )
        for store in vectorstores
    ]

    # init ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=retrievers,
        # weights=[0.25, 0.25, 0.25, 0.25]  # Adjust weights based on importance
    )

    # print("---RETRIEVING CONTEXT---")
    # return ensemble_retriever.invoke(question, k=3)

    print("--- 🌐 RETRIEVING AND RERANKING DYNAMIC CONTEXT ---")
    # Wrap your ensemble retriever with the compression retriever.
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    reranked_docs = compression_retriever.invoke(question)
    return reranked_docs

# --- Route 1: The GraphRAG Chain ---
# This chain is activated when the router classifies the question for GraphRAG.
graph_rag_chain = (# human printable format
    RunnablePassthrough.assign(context=lambda x: format_docs_with_metadata(retrieve_context(x["question"])))
    | analyst_prompt # system prompt, field-shots, user context formatting
    | ANSWER_LLM
)