"""
RAG Pipeline - orchestrates retrieval and generation.
Combines embeddings, vector search, and LLM generation.
Uses local ChromaDB for vector storage.
"""
from typing import List, Dict, Any
from autorag.rag.llm_client import LLMClient
from autorag.rag.embeddings import EmbeddingService
from autorag.rag.chroma_store import ChromaVectorStore
from autorag.utils.text_utils import chunk_documents


class RAGPipeline:
    """Complete RAG pipeline: retrieve relevant docs, generate answer."""
    
    def __init__(
        self,
        llm_provider: str,
        llm_api_key: str,
        llm_model: str = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        collection_name: str = "autorag"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            llm_provider: LLM provider (groq, openai, openrouter)
            llm_api_key: API key for the LLM provider
            llm_model: Optional model name (uses provider default if None)
            embedding_model: HuggingFace sentence-transformers model name
            chunk_size: Characters per chunk for document splitting
            chunk_overlap: Overlap characters between chunks
            collection_name: ChromaDB collection name
        """
        # Store chunking params for index_documents
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embedder = EmbeddingService(model_name=embedding_model)
        self.vector_store = ChromaVectorStore(collection_name=collection_name)
        self.llm_client = LLMClient(
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model
        )
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Chunk, embed, and store documents in vector database.
        
        Args:
            documents: List of dicts with 'id', 'text', 'metadata'
        """
        if not documents:
            return
        
        # Chunk using pipeline's configured params
        all_chunks = chunk_documents(
            documents, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in all_chunks]
        
        # Generate embeddings in batch
        embeddings = self.embedder.embed_batch(texts)
        
        # Store in ChromaDB
        self.vector_store.upsert_documents(all_chunks, embeddings)
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve (default: 5)
            temperature: LLM temperature (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            Dict with 'answer', 'sources', 'retrieved_docs'
        """
        # Step 1: Embed the question
        query_embedding = self.embedder.embed_text(question)
        
        # Step 2: Retrieve relevant documents
        retrieved_docs = self.vector_store.search(query_embedding, top_k=top_k)
        
        if not retrieved_docs:
            return {
                "answer": "No relevant documents found in the database.",
                "sources": [],
                "retrieved_docs": []
            }
        
        # Step 3: Build context from retrieved documents
        context = self._build_context(retrieved_docs)
        
        # Step 4: Generate answer using Groq
        answer = self._generate_answer(question, context, temperature)
        
        # Step 5: Format sources
        sources = [
            {
                "id": doc["id"],
                "score": doc["score"],
                "text": doc["text"][:200] + "..."  # Preview
            }
            for doc in retrieved_docs
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_docs": retrieved_docs
        }
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i}:\n{doc['text']}\n")
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str, temperature: float) -> str:
        """Generate answer using Groq LLM."""
        # Create prompt with context and question
        prompt = f"""You are a helpful assistant. Answer the question based on the context provided.

Context:
{context}

Question: {question}

Answer: Provide a clear, concise answer based only on the information in the context. If the context doesn't contain relevant information, say so."""

        # Call Groq API (model rotation handled by wrapper)
        response = self.llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return self.vector_store.get_stats()
    
    def clear_index(self):
        """Delete all vectors from index."""
        self.vector_store.delete_all()
