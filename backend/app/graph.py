from langgraph.graph import StateGraph, END
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.runnables import RunnablePassthrough
from setup.init import graph, EMBEDDINGS, create_vector_stores, ANSWER_LLM, RERANKER_MODEL, GraphState
from prompts.st_overflow import analyst_prompt
from utils.util import format_docs_with_metadata, escape_lucene_chars

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
LIMIT 100
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
        top_n=10  # This will return the top n most relevant documents.
    )

def retrieve_context(state):
    """
    Retrieve context from the ensemble retriever based on the user's question.
    """
    print("---RETRIEVING CONTEXT---")
    question = state["question"]
    
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

    vectorstores = [tagstore, userstore, questionstore, answerstore]

    retrievers = [
        store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=common_search_kwargs
        )
        for store in vectorstores
    ]

    ensemble_retriever = EnsembleRetriever(retrievers=retrievers)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    reranked_docs = compression_retriever.invoke(question)
    return {"context": reranked_docs}

def format_chat_history(state):
    """Format chat history for inclusion in the prompt."""
    print("---FORMATTING CHAT HISTORY---")
    chat_history = state["chat_history"]
    if not chat_history:
        return {"chat_history_formatted": ""}
    
    formatted_history = []
    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            formatted_history.append(f"User: {content}")
        elif role == "assistant":
            formatted_history.append(f"Assistant: {content}")
    
    return {"chat_history_formatted": "\n".join(formatted_history)}

def generate_answer(state):
    """
    Generate answer using the LLM.
    """
    print("---GENERATING ANSWER---")
    question = state["question"]
    context = state["context"]
    chat_history_formatted = state["chat_history"]

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs_with_metadata(context),
            chat_history_formatted=lambda x: chat_history_formatted
        )
        | analyst_prompt
        | ANSWER_LLM
    )
    
    answer = rag_chain.invoke({"question": question})
    return {"answer": answer}

# Define the graph
workflow = StateGraph(GraphState)

# Add the nodes
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("format_chat_history", format_chat_history)
workflow.add_node("generate_answer", generate_answer)

# Set the entrypoint
workflow.set_entry_point("retrieve_context")

# Add the edges
workflow.add_edge("retrieve_context", "format_chat_history")
workflow.add_edge("format_chat_history", "generate_answer")
workflow.add_edge("generate_answer", END)

# Compile the graph
graph_rag_chain = workflow.compile()
