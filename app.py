"""
RAG Chatbot - Streamlit Frontend
"""

import os
import uuid
import tempfile
import streamlit as st
from dotenv import load_dotenv

from database import (
    init_database,
    create_session,
    save_message,
    save_document,
    get_documents,
)
from rag_engine import (
    init_qdrant_collection,
    add_documents_to_qdrant,
    process_pdf,
    process_text_file,
    process_text,
    rag_query,
    get_collection_stats,
    delete_collection,
)

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 10px;
        border-radius: 10px;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


def initialize_app():
    """Initialize application state and databases."""
    if "initialized" not in st.session_state:
        try:
            # Try to initialize database (will fail gracefully if unavailable)
            db_ok = init_database()
            st.session_state.db_available = db_ok

            # Initialize Qdrant
            init_qdrant_collection()
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Initialization error: {e}")
            st.session_state.initialized = False
            st.session_state.db_available = False

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        if st.session_state.get("db_available", False):
            create_session(st.session_state.session_id, "New Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True


def sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.title("🤖 RAG Chatbot")
        st.markdown("---")

        # Document Upload Section
        st.subheader("📄 Upload Documents")

        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt"],
            help="Upload PDF or TXT files to add to the knowledge base"
        )

        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        # Process based on file type
                        if uploaded_file.name.endswith(".pdf"):
                            chunks = process_pdf(tmp_path)
                        else:
                            chunks = process_text_file(tmp_path)

                        # Add to Qdrant
                        num_chunks = add_documents_to_qdrant(chunks, uploaded_file.name)

                        # Save metadata to database if available
                        if st.session_state.get("db_available", False):
                            save_document(
                                uploaded_file.name,
                                uploaded_file.name.split(".")[-1],
                                num_chunks
                            )

                        # Cleanup
                        os.unlink(tmp_path)

                        st.success(f"Processed {num_chunks} chunks from {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing document: {e}")

        # Text input for direct knowledge
        st.markdown("---")
        st.subheader("📝 Add Text Knowledge")

        text_input = st.text_area(
            "Enter text to add to knowledge base",
            height=100,
            placeholder="Paste text here..."
        )

        if st.button("Add Text"):
            if text_input.strip():
                with st.spinner("Processing text..."):
                    try:
                        chunks = process_text(text_input)
                        num_chunks = add_documents_to_qdrant(chunks, "direct_input")
                        if st.session_state.get("db_available", False):
                            save_document("direct_text_input", "text", num_chunks)
                        st.success(f"Added {num_chunks} chunks to knowledge base")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter some text")

        # Collection Stats
        st.markdown("---")
        st.subheader("📊 Knowledge Base Stats")

        stats = get_collection_stats()
        st.metric("Total Chunks", stats.get("total_points", 0))
        st.text(f"Status: {stats.get('status', 'unknown')}")

        # Show uploaded documents (only if DB available)
        if st.session_state.get("db_available", False):
            try:
                documents = get_documents()
                if documents:
                    st.markdown("**Uploaded Documents:**")
                    for doc in documents[:5]:  # Show last 5
                        st.text(f"• {doc['filename']}")
            except Exception:
                pass

        # Settings
        st.markdown("---")
        st.subheader("⚙️ Settings")

        st.session_state.show_sources = st.checkbox(
            "Show sources",
            value=st.session_state.show_sources
        )

        # New Chat button
        if st.button("🔄 New Chat"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            create_session(st.session_state.session_id, "New Chat")
            st.rerun()

        # Danger zone
        with st.expander("⚠️ Danger Zone"):
            if st.button("Clear Knowledge Base", type="secondary"):
                if delete_collection():
                    init_qdrant_collection()
                    st.success("Knowledge base cleared")
                    st.rerun()
                else:
                    st.error("Failed to clear knowledge base")


def display_sources(sources):
    """Display source documents."""
    if sources and st.session_state.show_sources:
        with st.expander("📚 Sources", expanded=False):
            for i, source in enumerate(sources, 1):
                st.markdown(f"""
                **Source {i}** (Score: {source['score']:.3f})
                - File: `{source['source']}`
                - Content: {source['text'][:200]}...
                """)


def chat_interface():
    """Main chat interface."""
    st.header("💬 Chat with your Documents")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                display_sources(message["sources"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get chat history for context
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]  # Exclude current message
                    ]

                    # RAG query
                    response, sources = rag_query(prompt, history)

                    st.markdown(response)
                    display_sources(sources)

                    # Save to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })

                    # Save to database if available
                    if st.session_state.get("db_available", False):
                        save_message(st.session_state.session_id, "user", prompt)
                        save_message(st.session_state.session_id, "assistant", response)

                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def main():
    """Main application entry point."""
    initialize_app()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-openai-api-key-here":
        st.warning("Please set your OpenAI API key in the .env file")
        st.code("OPENAI_API_KEY=sk-your-key-here", language="bash")
        st.stop()

    # Show warning if database is unavailable
    if not st.session_state.get("db_available", False):
        st.warning("Database unavailable - chat history will only persist in current session")

    sidebar()
    chat_interface()


if __name__ == "__main__":
    main()
