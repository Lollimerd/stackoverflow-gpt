# app.py
import json, requests, datetime
import streamlit as st
import uuid # Used to generate unique IDs for chats
# from loader import render_page
from datetime import datetime
import streamlit.components.v1 as components
from streamlit_timeline import timeline 
from utils.utils import render_message_with_mermaid, display_container_name, get_system_config

# --- Page Configuration ---
st.set_page_config(page_title="Custom GPT", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# --- API Configuration ---
FASTAPI_URL = "http://0.0.0.0:8000/stream-ask"

# --- Initialize Session State for Multi-Chat ---
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "active_chat_id" not in st.session_state:
    # Create a default first chat on the first run
    first_chat_id = str(uuid.uuid4())
    st.session_state.active_chat_id = first_chat_id
    st.session_state.chats[first_chat_id] = {
        "title": "New Chat",
        "messages": [],
        "thoughts": [] # Retaining thoughts if you wish to use them per chat
    }

# Helper function to get the active chat object
def get_active_chat():
    return st.session_state.chats[st.session_state.active_chat_id]

# --- sidebar init ---
with st.sidebar:
    display_container_name()
    st.sidebar.title("⚙️ Settings", help="config settings here")
    # --- General Settings ---
    name = st.sidebar.text_input("Your Name", "test")

    # --- ADD THIS EXPANDER FOR SYSTEM DETAILS ---
    with st.expander("System Info & DB Details", expanded=False):
        config_data = get_system_config()
        if config_data:
            st.markdown(f"**Ollama Model:** `{config_data.get('ollama_model', 'N/A')}`")
            st.markdown(f"**Neo4j URL:** `{config_data.get('neo4j_url', 'N/A')}`")
            st.markdown(f"**DB connected:** `{config_data.get('container_name', 'N/A')}`")
            st.markdown(f"**Neo4j User:** `{config_data.get('neo4j_user', 'N/A')}`")
        else:
            st.error("Could not retrieve system info.")

    # --- Chat Management UI ---
    st.sidebar.subheader("Chats", help="navigate your chats here")
    if st.sidebar.button("➕ New Chat", use_container_width=True):
        new_chat_id = str(uuid.uuid4())
        st.session_state.chats[new_chat_id] = {
            "title": f"Chat {len(st.session_state.chats) + 1}",
            "messages": [],
            "thoughts": []
        }
        st.session_state.active_chat_id = new_chat_id
        st.rerun()

    # --- Display chats as individual buttons ---
    # Create a reversed list of chat IDs to display the newest chats first
    chat_ids = list(st.session_state.chats.keys())
    
    # Create a container with a fixed height inside the sidebar
    chat_container = st.container(height=150) # Adjust height as needed
    
    for chat_id in reversed(chat_ids):
        chat_data = st.session_state.chats[chat_id]
        
        # Use columns to place chat button and delete button on the same line
        col1, col2 = chat_container.columns([0.2, 0.8])
        
        with col1:
            # Button to delete the chat
            if st.button("🗑️", key=f"delete_chat_{chat_id}", use_container_width=True):
                # Remove the chat from the dictionary
                del st.session_state.chats[chat_id]
                
                # If the deleted chat was the active one, select a new active chat
                if st.session_state.active_chat_id == chat_id:
                    # Set to the first available chat, or None if no chats are left
                    remaining_chats = list(st.session_state.chats.keys())
                    st.session_state.active_chat_id = remaining_chats[0] if remaining_chats else None
                st.rerun()
        
        with col2:
            # Button to select the chat
            if st.button(
                chat_data['title'], 
                key=f"chat_button_{chat_id}", 
                use_container_width=True,
                disabled=(chat_id == st.session_state.active_chat_id)
            ):
                st.session_state.active_chat_id = chat_id
                st.rerun()

    # --- Clear history for the ACTIVE chat ---
    if st.button("Clear Active Chat History", use_container_width=True):
        active_chat = get_active_chat()
        active_chat["thoughts"] = []
        active_chat["messages"] = []
        st.rerun()    
    st.write("OPSEC ©LOLLIMERD 2025")
    # st.sidebar.markdown("---") # Visual separator

# Create tabs
# tab1, tab2, tab3, tab4= st.tabs(["💬OPSEC CHATBOT", "stackoverflow loader", "🗃📊 DEV LOGS", "📈💰📊Dashboard"])

active_chat = get_active_chat()
# f"{active_chat['title']}" # chat no
# --- UI Elements ---
st.title(f"🧠 Lollimerd's AI ")
st.header("""**:violet-badge[:material/star: OSU GPT]** **:blue-badge[:material/star: Ollama]** **:green-badge[:material/Verified: Mixture of Experts (MOE) model -> Qwen3]** **:blue-badge[:material/component_exchange: GraphRAG]**
""")
st.markdown("""Ask a question to get a real-time analysis from the knowledge graph. Feel free to ask the bot whatever your queries may be.
            Be specific in what you are asking, create table, generate graph of asking for data within a specified duration of time. Inferences, analysis and predictions are supported too :)
""")
st.subheader(body=f"Welcome back, {name} ⎛⎝ ≽  >  ⩊   < ≼ ⎠⎞")

# When displaying past messages from the ACTIVE chat:
for message in active_chat["messages"]:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            if "thought" in message and message["thought"]:
                with st.expander("Show Agent Thoughts"):
                    render_message_with_mermaid(message["thought"])
            render_message_with_mermaid(message["content"])
        else:
            st.markdown(message["content"])

# --- Main Interaction Logic (operates on the active chat) ---
if prompt := st.chat_input("Ask your question..."):
    # Append to the active chat's message list
    active_chat["messages"].append({"role": "user", "content": prompt})

    # Set a title for new chats based on the first message
    if active_chat["title"] == "New Chat" or active_chat["title"].startswith("Chat "):
        active_chat["title"] = prompt[:10] + "..." # Truncate for display

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing graph data..."):
            thought_expander = st.expander("Show Agent Thoughts")
            thought_placeholder = thought_expander.empty()
            answer_placeholder = st.empty()
            thought_content = ""
            answer_content = ""

            THINK_START_TAG = "<|THINK_START|>"
            THINK_END_TAG = "<|THINK_END|>"
            try:
                is_thinking = False
                buffer = ""
                
                with requests.post(FASTAPI_URL, json={"question": prompt}, stream=True) as r:
                    r.raise_for_status()
                    
                    for chunk in r.iter_content(chunk_size=32, decode_unicode=True):
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
                                    thought_content += buffer
                                    thought_placeholder.markdown(thought_content)
                                    buffer = ""
                                    break
                            else:
                                if THINK_START_TAG in buffer:
                                    answer_part, remainder = buffer.split(THINK_START_TAG, 1)
                                    answer_content += answer_part
                                    answer_placeholder.markdown(answer_content + " ▌")
                                    buffer = remainder
                                    is_thinking = True
                                else:
                                    answer_content += buffer
                                    answer_placeholder.markdown(answer_content + " ▌")
                                    buffer = ""
                                    break

                # --- Final Processing and Rendering ---
                render_message_with_mermaid(thought_content)
                render_message_with_mermaid(answer_content)

                # Append the final response to the active chat's message list
                active_chat["messages"].append({"role": "assistant", 
                                                "content": answer_content, 
                                                "thought": thought_content,
                                                "timestamp": datetime.now().strftime("%H:%M:%S")})
                st.rerun() # Rerun to update the chat list in the sidebar if the title changed

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API. Error: {e}")