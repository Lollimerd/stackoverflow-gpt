from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from typing import List
from prompts.core import analyst_prompt
from setup.init import NEO4J_PASSWORD, graph, EMBEDDINGS, NEO4J_URL, NEO4J_USERNAME, create_vector_stores, ANSWER_LLM
from utils.util import format_docs_with_metadata, escape_lucene_chars

# --- The core retrieval query for the RAG pipeline ---
test_query = """
// 1. Start with the entity node found by the vector search.
WITH node AS e, score AS vector_score

// 2. Entity Resolution.
OPTIONAL MATCH (e)-[:SAME_AS]->(canonical_entity)
WITH coalesce(canonical_entity, e) AS anchor_entity, vector_score

// 3. Get the Community ID (from Leiden clustering).
OPTIONAL MATCH (anchor_entity)
WHERE anchor_entity.communityId IS NOT NULL
WITH anchor_entity, vector_score, anchor_entity.communityId AS targetCommunityId

// 4. Find all entities within the same community.
MATCH (community_member)
WHERE community_member.communityId = targetCommunityId

// 5. Find all Sentence nodes connected to these community members.
MATCH (community_member)<-[]-(s:Sentence)
WITH s, anchor_entity, targetCommunityId, vector_score, collect(distinct community_member.name) as members

// 6. Perform a full-text search on these sentences.
CALL db.index.fulltext.queryNodes("Movie_devlev", $keyword_query) YIELD node, score AS keyword_score
WHERE node = s

// 7. Retrieve related Movie, Person, Category, Type, Country, Year nodes.
OPTIONAL MATCH (m)-[:IN_CATEGORY]->(cat:Category)
OPTIONAL MATCH (m)-[:TYPED_AS]->(t:Type)
OPTIONAL MATCH (m)-[:WHERE]->(c:Country)
OPTIONAL MATCH (m)-[:CREATED_ON]->(y:Year)
OPTIONAL MATCH (community_member:Person)-[:ACTED_IN|:DIRECTED]->(m)

// 8. Calculate combined score.
WITH s, m, cat, t, c, y, community_member, anchor_entity, targetCommunityId, vector_score, keyword_score,
     (vector_score * 0.5) + (keyword_score * 0.5) AS combined_score,
     collect(DISTINCT community_member.name) AS members

// 9. Return schema-matching output.
RETURN DISTINCT {movie_description: m.description} AS text, // fallback to movie description
    {id: m.id, // other movie properties
    title: m.title,
    duration: m.duration,
    listed_in: m.listed_in,
    year: m.year,
    release_year: m.release_year,
    rating: m.rating,
    day: m.day,
    month: m.month,
    date_str: m.date_str,

    // related node properties
    person: community_member.name,
    category: cat.name,
    type: t.type,
    country: c.name,
    year: y.value,

    // community properties
    communityId: targetCommunityId,
    membersInCommunity: members,
    vectorScore: vector_score,
    keywordScore: keyword_score,
    combinedScore: combined_score
  } AS metadata,
  combined_score AS score
ORDER BY score DESC
"""

# creating vector stores for different node labels
stores = create_vector_stores(graph, EMBEDDINGS, test_query)
movie_store = stores["moviestore"]
country_store = stores["countrystore"]
person_store = stores["personstore"]
category_store = stores["categorystore"]
country_store = stores["countrystore"]
type_store = stores["typestore"]

# ===========================================================================================================================================================
# Setting Up Retrievers from vectorstores for EnsembleRetriever 
# ===========================================================================================================================================================

# setting up retrievers from vectorstores with custom tailormade finetuning
def retrieve_context(question: str) -> List[Document]:
    """Retrieve context from the ensemble retriever based on the user's question."""
    
    # Define the common search arguments once
    common_search_kwargs = {
        'k': 10,
        'params': {
            'embedding': EMBEDDINGS.embed_query(question),
            "keyword_query": escape_lucene_chars(question), # for full text search
            },
        'fetch_k': 100,
        'score_threshold': 0.6,
        'lambda_mult': 0.5,
    }

    # Use a list of vectorstores
    vectorstores = [movie_store, person_store, category_store, country_store, type_store, person_store]

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

# --- The GraphRAG Chain ---
# This chain is activated when the router classifies the question for GraphRAG.
graph_rag_chain = (
    # human printable format
    RunnablePassthrough.assign(context=lambda x: format_docs_with_metadata(retrieve_context(x["question"])))
    | analyst_prompt # system prompt, field-shots, user context formatting
    | ANSWER_LLM # ollama qwen3:30b
)