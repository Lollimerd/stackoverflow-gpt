# app.py
import json
import requests
import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="Custom GPT", page_icon="🧠", layout="wide")

# --- API Configuration ---
FASTAPI_URL = "http://127.0.0.1:8000/stream-ask"

# --- UI Elements ---
st.title("🧠 CUSTOM AI")
st.markdown("Ask a question to get a real-time analysis from the knowledge graph. Feel free to ask it whatever your queries may be")
with st.sidebar:
    st.sidebar.title("⚙️ Settings")
    # genre = st.sidebar.selectbox("Genre", ["Action", "Comedy", "Drama"])
    name = st.sidebar.text_input("Your Name")
    st.divider()
    st.button("New Chat", use_container_width=True)
    st.button("Clear History", use_container_width=True)
st.subheader(body=f"Welcome back, {name}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
                # --- State variables for the frontend ---
                is_thinking = False
                buffer = ""
                
                with requests.post(FASTAPI_URL, json={"question": prompt}, stream=True) as r:
                    r.raise_for_status()
                    
                    # Use iter_content to get raw chunks as they arrive
                    for chunk in r.iter_content(chunk_size=256, decode_unicode=True):
                        buffer += chunk

                        # Process the buffer continuously
                        while THINK_START_TAG in buffer or THINK_END_TAG in buffer:
                            if is_thinking and THINK_END_TAG in buffer:
                                # If we are in a thought and find the end tag
                                thought_part, remainder = buffer.split(THINK_END_TAG, 1)
                                thought_content += thought_part
                                thought_placeholder.markdown(thought_content, unsafe_allow_html=True)
                                buffer = remainder
                                is_thinking = False
                            elif not is_thinking and THINK_START_TAG in buffer:
                                # If we are not in a thought and find the start tag
                                answer_part, remainder = buffer.split(THINK_START_TAG, 1)
                                answer_content += answer_part
                                answer_placeholder.markdown(answer_content + " ▌")
                                buffer = remainder
                                is_thinking = True
                            else:
                                # Break if a tag is split across chunks
                                break 
                        
                        # Process any remaining part of the buffer
                        if buffer:
                            if is_thinking:
                                thought_content += buffer
                                thought_placeholder.markdown(thought_content)
                            else:
                                answer_content += buffer
                                answer_placeholder.markdown(answer_content + " ▌")
                            buffer = ""
                        print(f"Buffer content: {buffer}")
                
                # Final update without the cursor
                answer_placeholder.markdown(answer_content)
                st.session_state.messages.append({"role": "assistant", "content": answer_content})

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API. Error: {e}")