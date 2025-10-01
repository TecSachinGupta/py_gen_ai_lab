import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from chatbot_backend import (
    chatbot, 
    retrieve_all_threads, 
    get_thread_preview,
    delete_thread
)

# =========================== Utilities ===========================
def generate_thread_id():
    """Generate a new UUID for thread identification."""
    return str(uuid.uuid4())

def reset_chat():
    """Create a new chat thread and reset the message history."""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    """Add a thread to the session state if it doesn't exist."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, thread_id)

def load_conversation(thread_id):
    """Load conversation messages from a specific thread."""
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        return state.values.get("messages", [])
    except:
        return []

def remove_thread(thread_id):
    """Remove a thread from session state and database."""
    if thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(thread_id)
    delete_thread(thread_id)

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# ============================ Configure page ============================
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        max-width: 100%;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #f7f7f8;
    }
    
    .assistant-message {
        background-color: #ffffff;
    }
    
    /* Sidebar styling */
    .sidebar .element-container {
        margin-bottom: 0.5rem;
    }
    
    div[data-testid="stSidebarNav"] {
        display: none;
    }
    
    /* Button styling for conversation list */
    .stButton button {
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Welcome screen */
    .welcome-container {
        text-align: center;
        padding: 4rem 2rem;
    }
    
    .welcome-container h1 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .welcome-container p {
        font-size: 1.2rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================ Sidebar ============================

with st.sidebar:
    st.title("🤖 AI Assistant")
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        reset_chat()
        st.rerun()
    
    st.divider()
    
    st.subheader("Conversations")

    # Display conversations with delete buttons
    if st.session_state["chat_threads"]:
        for thread_id in st.session_state["chat_threads"]:
            col1, col2 = st.columns([5, 1])
            
            with col1:
                # Get preview text for the conversation
                preview_text = get_thread_preview(thread_id)
                
                button_type = "primary" if thread_id == st.session_state["thread_id"] else "secondary"
                
                if st.button(
                    preview_text, 
                    key=f"conv_{thread_id}",
                    use_container_width=True,
                    type=button_type if thread_id == st.session_state["thread_id"] else "secondary"
                ):
                    st.session_state["thread_id"] = thread_id
                    messages = load_conversation(thread_id)
                    
                    # Convert messages to display format
                    temp_messages = []
                    for msg in messages:
                        if isinstance(msg, HumanMessage):
                            temp_messages.append({"role": "user", "content": msg.content})
                        elif isinstance(msg, AIMessage):
                            temp_messages.append({"role": "assistant", "content": msg.content})
                    
                    st.session_state["message_history"] = temp_messages
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"del_{thread_id}"):
                    # If deleting current conversation, reset to new chat
                    if thread_id == st.session_state["thread_id"]:
                        reset_chat()
                    else:
                        remove_thread(thread_id)
                    st.rerun()
    else:
        st.info("No conversations yet. Start a new chat!")
    
    st.divider()
    st.caption("Built with LangGraph + Streamlit")

# ============================ Main UI ============================

# Show welcome screen if no messages
if not st.session_state["message_history"]:
    st.markdown("""
    <div class='welcome-container'>
        <h1>👋 Welcome to AI Assistant</h1>
        <p>I can help you with web searches, calculations, and stock prices!</p>
        <p style='margin-top: 2rem; color: #999;'>Start typing below to begin...</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Render message history
    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Type your message here..."):
    # Add user message to history
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Configuration for LangGraph
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }
    
    # Assistant response with streaming
    with st.chat_message("assistant"):
        status_holder = {"box": None}
        
        def ai_stream_generator():
            """Generator that streams AI responses and handles tool calls."""
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Handle tool messages
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}`...", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`...",
                            state="running",
                            expanded=True,
                        )
                
                # Stream AI message content
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
        
        # Stream and collect the response
        ai_message = st.write_stream(ai_stream_generator())
        
        # Finalize tool status if any tool was used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", 
                state="complete", 
                expanded=False
            )
    
    # Save assistant message to history
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
    
    # Update thread list if this is a new thread
    if st.session_state["thread_id"] not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, st.session_state["thread_id"])