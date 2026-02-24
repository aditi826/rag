"""
RAG Engine module for document processing and retrieval.
Uses OpenAI for embeddings/LLM and Qdrant for vector storage.
"""

import os
from typing import List
from dotenv import load_dotenv

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
LLM_MODEL = "gpt-4o-mini"


def init_qdrant_collection():
    """Initialize Qdrant collection if it doesn't exist."""
    collections = qdrant_client.get_collections().collections
    collection_names = [c.name for c in collections]

    if COLLECTION_NAME not in collection_names:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            ),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection {COLLECTION_NAME} already exists")


def get_embedding(text: str) -> List[float]:
    """Generate embedding for a text using OpenAI."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)


def process_pdf(file_path: str) -> List[str]:
    """Load and process a PDF file."""
    reader = PdfReader(file_path)

    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    chunks = chunk_text(full_text)
    return chunks


def process_text_file(file_path: str) -> List[str]:
    """Load and process a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    chunks = chunk_text(full_text)
    return chunks


def process_text(text: str) -> List[str]:
    """Process raw text input."""
    return chunk_text(text)


def add_documents_to_qdrant(chunks: List[str], source: str) -> int:
    """Add document chunks to Qdrant collection."""
    init_qdrant_collection()

    # Get current point count for ID offset
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    id_offset = collection_info.points_count

    points = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)

        point = PointStruct(
            id=id_offset + i,
            vector=embedding,
            payload={
                "text": chunk,
                "source": source,
                "chunk_index": i,
            }
        )
        points.append(point)

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )

    return len(chunks)


def search_documents(query: str, top_k: int = 5) -> List[dict]:
    """Search for relevant documents in Qdrant."""
    try:
        query_embedding = get_embedding(query)

        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
        )

        documents = []
        for result in results:
            documents.append({
                "text": result.payload.get("text", ""),
                "source": result.payload.get("source", "unknown"),
                "score": result.score,
            })

        return documents
    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_response(
    query: str,
    context_docs: List[dict],
    chat_history: List[dict] = None,
    system_prompt: str = None
) -> str:
    """Generate a response using OpenAI with RAG context."""

    # Build context from retrieved documents
    if context_docs:
        context = "\n\n---\n\n".join([
            f"Source: {doc['source']}\n{doc['text']}"
            for doc in context_docs
        ])
    else:
        context = "No relevant documents found."

    # Default system prompt
    if system_prompt is None:
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.

Instructions:
- Use the provided context to answer questions accurately
- If the context doesn't contain relevant information, say so honestly
- Be concise and helpful in your responses
- Cite sources when possible"""

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context from documents:\n\n{context}"},
    ]

    # Add chat history
    if chat_history:
        for msg in chat_history[-10:]:  # Last 10 messages for context
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    # Add current query
    messages.append({"role": "user", "content": query})

    # Generate response
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
    )

    return response.choices[0].message.content


def rag_query(query: str, chat_history: List[dict] = None) -> tuple[str, List[dict]]:
    """
    Main RAG query function.
    Returns the response and the retrieved documents.
    """
    # Search for relevant documents
    relevant_docs = search_documents(query, top_k=5)

    # Generate response with context
    response = generate_response(
        query=query,
        context_docs=relevant_docs,
        chat_history=chat_history
    )

    return response, relevant_docs


def get_collection_stats() -> dict:
    """Get statistics about the Qdrant collection."""
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "total_points": collection_info.points_count,
            "status": collection_info.status,
        }
    except Exception:
        return {"total_points": 0, "status": "not_initialized"}


def delete_collection():
    """Delete the Qdrant collection (use with caution)."""
    try:
        qdrant_client.delete_collection(COLLECTION_NAME)
        return True
    except Exception as e:
        print(f"Error deleting collection: {e}")
        return False
