from langchain_core.documents import Document
from typing import List
import json, docker

# --- Dynamic Container Discovery ---
def find_container_by_port(port: int) -> str:
    """Inspects running Docker containers to find which one is using the specified port."""
    if not port:
        return "Invalid port"
    try:
        # Connect to the Docker daemon
        client = docker.from_env()
        containers = client.containers.list()

        for container in containers:
            # The .ports attribute is a dictionary like: {'7687/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '7687'}]}
            port_mappings = container.ports
            for container_port, host_mappings in port_mappings.items():
                if host_mappings:
                    for mapping in host_mappings:
                        if mapping.get("HostPort") == str(port):
                            return container.name # Found it!

        return "No matching container found"
    except docker.errors.DockerException:
        return "Docker daemon not running or not accessible"
    except Exception as e:
        return f"An error occurred: {e}"
    
def format_docs_with_metadata(docs: list[Document]) -> str:
    """Formats documents into a single string for the context."""
    formatted_blocks = []
    for doc in docs:
        # Format metadata as key: value pairs, one per line
        metadata_items = [f"{key}: {value}" for key, value in doc.metadata.items()]
        metadata_str = "\n".join(metadata_items)

        # Create a combined block for the document's content and its metadata
        block = (
            f"\n--------- CONTENT ---------\n"
            f"{doc.page_content}\n"
            f"--------- METADATA ---------\n"
            f"{metadata_str}"
        )
        formatted_blocks.append(block)

    # Combine all the individual document blocks with a clear separator
    final_context_str = "\n\n" + "=" * 100 + "\n\n".join(formatted_blocks)

    # --- ✨ NEW: Added print statements for debugging retrieved context ---
    print("\n" + "=" * 100)
    print("--- 📄 RETRIEVED CONTEXT FOR LLM ---")
    print(final_context_str)
    print(f"\n--- 📊 Documents retrieved: {len(docs)} ---")
    print("=" * 100 + "\n")
    return final_context_str