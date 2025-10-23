# app.py
import json, requests
import streamlit as st
from loader import render_page

# --- Page Configuration ---
st.set_page_config(page_title="Custom GPT", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Custom Bot", "Stackoverflow loader", "Tab 3"])

with tab1:
    with st.sidebar:
        st.sidebar.title("⚙️ Settings")
        genre = st.sidebar.selectbox("Genre", ["Action", "Comedy", "Drama"])
        name = st.sidebar.text_input("Your Name", "test")
        st.divider()
        if st.button("Clear History", use_container_width=True):
            st.session_state.thoughts = []
            st.session_state.messages = []
            st.rerun()  
    st.subheader(body=f"Welcome back, {name}")  

    # --- API Configuration ---
    FASTAPI_URL = "http://0.0.0.0:8080/stream-ask"

    # --- UI Elements ---
    st.title("🧠 CUSTOM AI")
    st.badge("stackoverflow", icon="🧊")
    st.markdown("Ask a question to get a real-time analysis from the knowledge graph. Feel free to ask it whatever your queries may be")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "thoughts" not in st.session_state:
        st.session_state.thoughts = []

    # When displaying past messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # 1. Display the thought first (if available)
                if "thought" in message and message["thought"]:
                    with st.expander("Show Agent Thoughts"):
                        st.markdown(message["thought"])
                
                # 2. Then display the main answer content
                st.markdown(message["content"])
            else:
                # For user messages, simply display their content
                st.markdown(message["content"])

with tab2:
    render_page()

st.divider()
# --- Main Interaction Logic ---
if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing graph data..."):
            # **MODIFIED**: Set up placeholders for thoughts and the final answer
            thought_expander = st.expander("Show Agent Thoughts")
            thought_placeholder = thought_expander.empty()
            answer_placeholder = st.empty()
            thought_content = ""
            answer_content = ""

            # --- Define the same tags used in the backend ---
            THINK_START_TAG = "<|THINK_START|>"
            THINK_END_TAG = "<|THINK_END|>"
            try:
                is_thinking = False
                buffer = ""
                
                with requests.post(FASTAPI_URL, json={"question": prompt}, stream=True) as r:
                    r.raise_for_status()
                    
                    for chunk in r.iter_content(chunk_size=256, decode_unicode=True):
                        buffer += chunk
                        
                        while True:
                            if is_thinking:
                                if THINK_END_TAG in buffer:
                                    thought_part, remainder = buffer.split(THINK_END_TAG, 1)
                                    thought_content += thought_part
                                    thought_placeholder.markdown(thought_content)
                                    buffer = remainder
                                    is_thinking = False
                                else:
                                    # No end tag yet, all current buffer is part of thought
                                    thought_content += buffer
                                    thought_placeholder.markdown(thought_content)
                                    buffer = "" # Processed all current buffer content as thought
                                    break # No more tag changes in this chunk, wait for next chunk
                            else: # not is_thinking
                                if THINK_START_TAG in buffer:
                                    answer_part, remainder = buffer.split(THINK_START_TAG, 1)
                                    answer_content += answer_part
                                    answer_placeholder.markdown(answer_content + " ▌")
                                    buffer = remainder
                                    is_thinking = True
                                else:
                                    # No start tag yet, all current buffer is part of answer
                                    answer_content += buffer
                                    answer_placeholder.markdown(answer_content + " ▌")
                                    buffer = "" # Processed all current buffer content as answer
                                    break # No more tag changes in this chunk, wait for next chunk
                
                # Final update without the cursor
                answer_placeholder.markdown(answer_content)
                st.session_state.messages.append({"role": "assistant", "content": answer_content, "thought": thought_content})

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API. Error: {e}")