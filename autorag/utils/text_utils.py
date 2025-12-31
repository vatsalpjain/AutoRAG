"""
Text utilities for AutoRAG.
Shared chunking logic used by both pipeline and CLI.
"""
import random
from typing import List, Dict, Any


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks for better RAG retrieval.
    
    Args:
        text: Full document text
        doc_id: Source document ID for metadata
        chunk_size: Characters per chunk (default: 500)
        chunk_overlap: Overlap between chunks (default: 50)
        
    Returns:
        List of chunk documents with 'id', 'text', 'metadata'
    """
    chunks = []
    text = text.strip()
    
    if len(text) <= chunk_size:
        return [{"id": f"{doc_id}_0", "text": text, "metadata": {"source": doc_id, "chunk": 0}}]
    
    start = 0
    chunk_idx = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            search_start = max(end - 100, start)
            best_break = max(
                text.rfind('.', search_start, end),
                text.rfind('?', search_start, end),
                text.rfind('!', search_start, end)
            )
            if best_break > search_start:
                end = best_break + 1
        
        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append({
                "id": f"{doc_id}_{chunk_idx}",
                "text": chunk_text_str,
                "metadata": {"source": doc_id, "chunk": chunk_idx}
            })
            chunk_idx += 1
        
        start = end - chunk_overlap
        if start >= len(text) - chunk_overlap:
            break
    
    return chunks


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Chunk multiple documents.
    
    Args:
        documents: List of raw documents with 'id', 'text', 'metadata'
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of all chunks from all documents
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"], doc["id"], chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
    return all_chunks


def sample_chunks_for_qa(
    chunks: List[Dict[str, Any]],
    target_count: int,
    questions_per_chunk: int = 1
) -> List[Dict[str, Any]]:
    """
    Randomly sample chunks for Q&A generation to maximize diversity.
    
    Args:
        chunks: List of all chunks
        target_count: Target number of Q&A pairs to generate
        questions_per_chunk: Questions to generate per chunk (default: 1)
        
    Returns:
        Randomly sampled subset of chunks
    """
    # Calculate how many chunks we need
    chunks_needed = min(len(chunks), (target_count // questions_per_chunk) + 1)
    
    # Random sample for diversity
    if len(chunks) <= chunks_needed:
        return chunks
    
    return random.sample(chunks, chunks_needed)
