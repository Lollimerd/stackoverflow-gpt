class BaseLogger:
    def __init__(self) -> None:
        self.info = print

def extract_title_and_question(input_string):
    lines = input_string.strip().split("\n")

    title = ""
    question = ""
    is_question = False  # flag to know if we are inside a "Question" block

    for line in lines:
        if line.startswith("Title:"):
            title = line.split("Title: ", 1)[1].strip()
        elif line.startswith("Question:"):
            question = line.split("Question: ", 1)[1].strip()
            is_question = (
                True  # set the flag to True once we encounter a "Question:" line
            )
        elif is_question:
            # if the line does not start with "Question:" but we are inside a "Question" block,
            # then it is a continuation of the question
            question += "\n" + line.strip()

    return title, question


def create_vector_index(driver) -> None:
    index_query = "CREATE VECTOR INDEX stackoverflow IF NOT EXISTS FOR (m:Question) ON m.embedding"
    try:
        driver.query(index_query)
    except:  # Already exists
        pass
    index_query = (
        "CREATE VECTOR INDEX top_answers IF NOT EXISTS FOR (m:Answer) ON m.embedding"
    )
    try:
        driver.query(index_query)
    except:  # Already exists
        pass


def create_constraints(driver):
    driver.query(
        "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE (q.id) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT answer_id IF NOT EXISTS FOR (a:Answer) REQUIRE (a.id) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE (u.id) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE (t.name) IS UNIQUE"
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

import json
from typing import List

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
            f"Content: \n{doc.page_content}\n"
            f"--- METADATA ---\n"
            f"{metadata_str}"
        )
        formatted_blocks.append(block)

    # Join all the individual document blocks with a clear separator
    return "\n\n======================================================\n\n".join(formatted_blocks)