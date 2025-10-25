from setup.init import graph, EMBEDDINGS, create_vector_stores, ANSWER_LLM, compressor, RERANKER_MODEL
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from typing import List, Dict
from langchain_core.documents import Document
from prompts.st_overflow import analyst_prompt
from langchain_core.tools import Tool
from langchain_core.runnables import RunnablePassthrough
from utils.util import format_docs_with_metadata, escape_lucene_chars
from langchain_core.messages import HumanMessage, AIMessage

# ===========================================================================================================================================================
# Crafting custom cypher retrieval queries
# ===========================================================================================================================================================
retrieval_query = """
// Start from vector search result variables: `node`, `score`
WITH node, score
// Route any node type to related Question(s) via UNION branches to avoid implicit grouping
CALL {
  WITH node
  // If node is a Question, use it directly
  WITH node
  MATCH (q:Question)
  WHERE node:Question AND id(q) = id(node)
  RETURN q
  UNION
  // If node is an Answer, route to its Question
  WITH node
  MATCH (node:Answer)-[:ANSWERS]->(q:Question)
  RETURN q
  UNION
  // If node is a Tag, route to Questions tagged with it
  WITH node
  MATCH (q:Question)-[:TAGGED]->(node:Tag)
  RETURN q
  UNION
  // If node is a User, include Questions they asked
  WITH node
  MATCH (node:User)-[:ASKED]->(q:Question)
  RETURN q
  UNION
  // If node is a User, include Questions they answered
  WITH node
  MATCH (node:User)-[:PROVIDED]->(:Answer)-[:ANSWERS]->(q:Question)
  RETURN q
}
WITH DISTINCT q AS question, node, score

// Community detection: compute overlap and optionally filter to same community when available
WITH 
  question, 
  node, 
  score, 
  any(x IN coalesce(question.communityId, []) WHERE x IN coalesce(node.communityId, [])) AS sameCommunity,
  (size(coalesce(question.communityId, [])) > 0 AND size(coalesce(node.communityId, [])) > 0) AS bothHaveCommunity
WHERE NOT bothHaveCommunity OR sameCommunity

// Build rich context for each question
// Core question data
WITH DISTINCT question, score, sameCommunity,
     coalesce(question.communityId, []) AS qComm,
     coalesce(node.communityId, []) AS nComm,
     {
  id: question.id,
  title: question.title,
  body: question.body,
  link: question.link,
  score: question.score,
  favorite_count: question.favorite_count,
  creation_date: toString(question.creation_date)
} AS questionDetails

// Askers
OPTIONAL MATCH (asker:User)-[:ASKED]->(question)
WITH question, score, sameCommunity, qComm, nComm, questionDetails, {
  id: asker.id,
  display_name: asker.display_name,
  reputation: asker.reputation
} AS askerDetails

// Tags
OPTIONAL MATCH (question)-[:TAGGED]->(tag:Tag)
WITH question, score, sameCommunity, qComm, nComm, questionDetails, askerDetails,
     COLLECT(DISTINCT tag.name) AS tags

// Answers + providers
OPTIONAL MATCH (answer:Answer)-[:ANSWERS]->(question)
OPTIONAL MATCH (provider:User)-[:PROVIDED]->(answer)
WITH question, score, sameCommunity, qComm, nComm, questionDetails, askerDetails, tags,
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

// Final projection
RETURN
  'Title: ' + coalesce(question.title, '') + '\nBody: ' + coalesce(question.body, '') AS text,
  {
    question_details: questionDetails,
    asked_by: askerDetails,
    tags: tags,
    answers: {
      answers: answers
    },
    community: {
      questionCommunityId: qComm,
      nodeCommunityId: nComm,
      sameCommunity: sameCommunity
    },
    simscore: score
  } AS metadata,
  score
ORDER BY score DESC
LIMIT 500
"""

# Create vector stores
stores = create_vector_stores(graph, EMBEDDINGS, retrieval_query)
tagstore = stores['tagstore']
userstore = stores['userstore']
questionstore = stores['questionstore']
answerstore = stores['answerstore']

# create compressor
compressor = CrossEncoderReranker(
        model=RERANKER_MODEL,
        top_n=20  # This will return the top n most relevant documents.
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
        'k': 10,
        'params': {
            'embedding': EMBEDDINGS.embed_query(question),
            'keyword_query': escape_lucene_chars(question)
            },
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

def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Format chat history for inclusion in the prompt."""
    if not chat_history:
        return ""
    
    formatted_history = []
    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            formatted_history.append(f"User: {content}")
        elif role == "assistant":
            # Only include the main content, not the thought process
            formatted_history.append(f"Assistant: {content}")
    
    return "\n".join(formatted_history)

# --- Route 1: The GraphRAG Chain ---
# This chain is activated when the router classifies the question for GraphRAG.
graph_rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs_with_metadata(retrieve_context(x["question"])),
        chat_history_formatted=lambda x: format_chat_history(x.get("chat_history", []))
    )
    | analyst_prompt
    | ANSWER_LLM
)