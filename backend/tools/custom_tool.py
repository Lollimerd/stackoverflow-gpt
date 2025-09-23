from setup.init import graph, EMBEDDINGS, create_vector_stores
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from typing import List
from langchain_core.documents import Document

# ===========================================================================================================================================================
# Crafting custom cypher retrieval queries
# ===========================================================================================================================================================

retrieval_query = """
// 1. Start with the node found by the vector search (passed in by LangChain as `node` and `score`).
WITH node, score

// 2. Get the Community ID from the hit node. The schema indicates it's a list, so we take the first element.
WITH node.communityId[0] AS targetCommunityId, score

// 3. For each community identified, find the highest vector similarity score among the hits.
// This ensures we retrieve the context for each relevant community only once, using its best score.
WITH targetCommunityId, max(score) AS topScore

// 4. Find the main Question node associated with that community. This becomes our context anchor.
MATCH (question:Question)
WHERE question.communityId[0] = targetCommunityId

// 5. Fetch the user who asked the question.
OPTIONAL MATCH (asker:User)-[:ASKED]->(question)

// 6. Collect all tags associated with the question.
OPTIONAL MATCH (question)-[:TAGGED]->(tag:Tag)
WITH question, topScore, asker, COLLECT(DISTINCT tag.name) AS tags

// 7. Collect all answers and the users who provided them.
OPTIONAL MATCH (answer:Answer)-[:ANSWERS]->(question)
OPTIONAL MATCH (provider:User)-[:PROVIDED]->(answer)
WITH question, topScore, asker, tags, COLLECT(DISTINCT {
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

// 8. Format and return the final results for the LLM.
RETURN
    // The primary text content for retrieval (the question's title and body).
    'Title: ' + question.title + '\nBody: ' + question.body AS text,
    
    // The structured metadata containing the full graph context for that question.
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
        simscore: topScore // Pass along the highest score for this community.
    } AS metadata,
    
    // Return the score itself for sorting.
    topScore as score
ORDER BY score DESC
"""

# Create vector stores
stores = create_vector_stores(graph, EMBEDDINGS, retrieval_query)
tagstore = stores['tagstore']
userstore = stores['userstore']
questionstore = stores['questionstore']
answerstore = stores['answerstore']

# ===========================================================================================================================================================
# Setting Up Retrievers from vectorstores for EnsembleRetriever 
# ===========================================================================================================================================================

# setting up retrievers from vectorstores with custom tailormade finetuning
def retrieve_context(question: str) -> List[Document]:
    """Retrieve context from the ensemble retriever based on the user's question."""
    
    # Define the common search arguments once
    common_search_kwargs = {
        'k': 10,
        'params': {'embedding': EMBEDDINGS.embed_query(question)},
        'fetch_k': 100,
        'score_threshold': 0.6,
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

    ensemble_retriever = EnsembleRetriever(
        retrievers=retrievers
    )

    print("---RETRIEVING CONTEXT---")
    # return vectorstore.similarity_search(question, k=6)
    return ensemble_retriever.invoke(question, k=3)