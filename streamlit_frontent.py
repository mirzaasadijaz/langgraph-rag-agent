import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk
from chatbot_backend import chatbot, memory 
from tools import process_pdf_from_frontend
from dotenv import load_dotenv

load_dotenv()

def retrieve_all_threads():
    all_threads = []
    for checkpoint in memory.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        if thread_id not in all_threads:
            all_threads.append(thread_id)
    return list(all_threads)

def init_session_state():
    if 'chat_threads' not in st.session_state:
        db_threads = retrieve_all_threads()
        
        if not db_threads:
            db_threads = [str(uuid.uuid4())]
            
        st.session_state['chat_threads'] = db_threads
        st.session_state['current_thread'] = db_threads[0]

def render_sidebar():
    with st.sidebar:
        st.title("💬 Conversations")
        
        # ==========================================
        # PDF UPLOAD SECTION
        # ==========================================
        st.divider()
        st.write("**📄 Document Q&A**")
        uploaded_pdf = st.file_uploader("Upload a PDF to query", type=["pdf"])
        
        if uploaded_pdf is not None:
            if st.session_state.get('last_uploaded_file') != uploaded_pdf.name:
                with st.spinner("Indexing PDF for the agent..."):
                    process_pdf_from_frontend(uploaded_pdf.getvalue())
                    st.session_state['last_uploaded_file'] = uploaded_pdf.name
                st.success(f"{uploaded_pdf.name} indexed! The AI can now search it.")
                
        st.divider()
        
        if st.button("➕ New Chat", use_container_width=True):
            new_thread = str(uuid.uuid4())
            st.session_state['chat_threads'].insert(0, new_thread)
            st.session_state['current_thread'] = new_thread
            st.rerun()
            
        st.divider()
        st.write("**Previous Chats**")
        
        for thread_id in st.session_state['chat_threads']:
            config = {
                'configurable': {
                    'thread_id': thread_id  
                },
                'metadata': {
                    'thread_id': thread_id, 
                    'source': 'streamlit_frontend'
                },
                'run_name': 'My_App_Chat_Turn' 
            }
            state = chatbot.get_state(config)
            
            if state and hasattr(state, 'values') and state.values and "messages" in state.values and len(state.values["messages"]) > 0:
                first_msg = state.values["messages"][0].content
                chat_label = first_msg[:25] + "..."
            else:
                chat_label = "Empty Chat"
            
            if st.button(chat_label, key=thread_id, use_container_width=True):
                st.session_state['current_thread'] = thread_id
                st.rerun()

def stream_response(user_input: str, config: dict, status_container):
    stream = chatbot.stream(
        {'messages': [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages" 
    )
    
    for msg_chunk, _ in stream:
        if hasattr(msg_chunk, 'tool_call_chunks') and msg_chunk.tool_call_chunks:
            for tool_call in msg_chunk.tool_call_chunks:
                if 'name' in tool_call and tool_call['name']:
                    status_container.write(f"🛠️ Calling tool: `{tool_call['name']}`...")

        if type(msg_chunk).__name__ == "ToolMessage":
            status_container.write(f"✅ Tool `{msg_chunk.name}` returned results.")

        if isinstance(msg_chunk, (AIMessageChunk, AIMessage)):
            if hasattr(msg_chunk, 'content') and msg_chunk.content:
                yield msg_chunk.content

def main():
    st.title("My AI Agent (with Tools)")
    
    init_session_state()
    render_sidebar()

    active_thread = st.session_state['current_thread']
    config = {
        'configurable': {'thread_id': active_thread},
        'metadata': {'thread_id': active_thread, 'source': 'streamlit_frontend'},
        'run_name': 'My_App_Chat_Turn' 
    }

    state = chatbot.get_state(config)
    messages = state.values.get("messages", []) if state and hasattr(state, 'values') and state.values else []

    # Display Chat History (Filtering out raw ToolMessages so the UI stays clean)
    for msg in messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # Handle New Input
    if user_input := st.chat_input('Type here...'):
        with st.chat_message('user'):
            st.markdown(user_input)

        with st.chat_message('assistant'):
            status_container = st.status("🤖 Agent thinking...", expanded=True)
            st.write_stream(stream_response(user_input, config, status_container))
            status_container.update(label="✅ Response complete", state="complete", expanded=False)

if __name__ == "__main__":
    main()