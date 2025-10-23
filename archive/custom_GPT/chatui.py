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