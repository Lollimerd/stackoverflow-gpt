# Replicating OpenWebUI with Streamlit

OpenWebUI is a popular web-based interface for interacting with local LLMs. Here's how you can create a similar interface using Streamlit with features like chat, model selection, and conversation history.

## Core Features Implementation

### 1. Basic Chat Interface
```python
import streamlit as st
from datetime import datetime

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = "llama3"

# Page config
st.set_page_config(
    page_title="OpenWebUI-like Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Model selection
    st.session_state.model = st.selectbox(
        "Select Model",
        ["llama3", "mistral", "phi3", "gemma"],
        index=0
    )
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant.",
        height=100
    )
    
    # Parameters
    with st.expander("Model Parameters"):
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
        max_tokens = st.slider("Max Tokens", 1, 4096, 512)
    
    st.divider()
    st.button("New Chat", use_container_width=True)
    st.button("Clear History", use_container_width=True)

# Main chat area
st.title(f"💬 Chat with {st.session_state.model.capitalize()}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar", None)):
        st.markdown(message["content"])
        if "timestamp" in message:
            st.caption(message["timestamp"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message to chat history
    user_msg = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "avatar": "👤"
    }
    st.session_state.messages.append(user_msg)
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        st.caption(user_msg["timestamp"])
    
    # Generate assistant response (placeholder - replace with actual LLM call)
    assistant_response = f"This is a simulated response from {st.session_state.model}. In a real implementation, this would connect to your LLM."
    
    # Add assistant response to chat history
    assistant_msg = {
        "role": "assistant",
        "content": assistant_response,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "avatar": "🤖"
    }
    st.session_state.messages.append(assistant_msg)
    
    # Display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(assistant_response)
        st.caption(assistant_msg["timestamp"])
```

### 2. Model Management System
```python
# Add this to your sidebar
with st.sidebar:
    st.subheader("Model Management")
    
    # Model download/selection
    with st.expander("Download Models"):
        selected_model = st.selectbox(
            "Available Models",
            ["llama3-8b", "mistral-7b", "phi3-mini", "gemma-2b"]
        )
        if st.button(f"Download {selected_model}"):
            st.toast(f"Downloading {selected_model}...", icon="⏳")
            # Add your actual download logic here
    
    # Model status
    st.divider()
    st.subheader("Model Status")
    st.progress(75, text="Memory Usage")
    st.metric("VRAM Used", "8.2/12 GB")
```

### 3. Conversation History
```python
# Add this to your sidebar
with st.sidebar:
    st.subheader("Conversation History")
    
    # Sample conversation history
    conversations = [
        "Today's chat",
        "Python help",
        "Project ideas",
        "Research assistance"
    ]
    
    for conv in conversations:
        st.button(
            conv,
            use_container_width=True,
            key=f"conv_{conv}"
        )
```

### 4. Advanced Features

#### a. Document Upload and Context
```python
# Add to sidebar
with st.sidebar:
    st.subheader("Document Context")
    uploaded_files = st.file_uploader(
        "Upload documents for context",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} document(s) uploaded")
        # Add your document processing logic here
```

#### b. RAG Configuration
```python
# Add to sidebar parameters
with st.sidebar.expander("RAG Parameters"):
    rag_enabled = st.toggle("Enable RAG", True)
    if rag_enabled:
        st.selectbox("Retriever", ["BM25", "Vector", "Hybrid"])
        st.slider("Context Chunks", 1, 10, 3)
        st.slider("Similarity Threshold", 0.0, 1.0, 0.7)
```

## Complete Implementation Notes

1. **For Actual LLM Integration**:
   - Replace the placeholder response with calls to your LLM (Ollama, vLLM, etc.)
   - Use libraries like `requests` to call local API endpoints

2. **For Production Use**:
   - Add authentication with `st.experimental_user`
   - Implement proper session management
   - Add error handling for model calls

3. **Enhancements**:
   - Add markdown support in chat
   - Implement code syntax highlighting
   - Add model streaming for token-by-token responses

4. **Styling**:
   - Use custom CSS with `st.markdown` for better visuals
   - Consider using `st.columns` for more complex layouts

This implementation provides the core functionality of OpenWebUI in a Streamlit interface. You can extend it further based on your specific requirements.