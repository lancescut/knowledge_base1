"""
Chat endpoints for RAG queries.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import json
import asyncio
from loguru import logger

router = APIRouter()


class Message(BaseModel):
    """Chat message model."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow()
        super().__init__(**data)


class ChatRequest(BaseModel):
    """Chat request model."""
    messages: List[Message]
    session_id: Optional[str] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = True


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    session_id: str
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    timestamp: datetime


class SessionInfo(BaseModel):
    """Chat session information."""
    session_id: str
    title: str
    created_at: datetime
    last_updated: datetime
    message_count: int


class SessionHistory(BaseModel):
    """Complete session history."""
    session_id: str
    title: str
    messages: List[Message]
    created_at: datetime
    last_updated: datetime


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for RAG queries."""
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get the latest user message
    if not request.messages or request.messages[-1].role != "user":
        raise HTTPException(
            status_code=400,
            detail="Request must contain at least one user message"
        )
    
    user_query = request.messages[-1].content
    
    try:
        # Import services here to avoid circular imports
        from services.embedding_service import embedding_service
        from services.generation_service import generation_service
        from services.chromadb_service import chromadb_service
        
        import time
        start_time = time.time()
        
        # 1. Generate embeddings for user query
        retrieval_start = time.time()
        query_embedding = await embedding_service.encode([user_query])
        
        # 2. Retrieve relevant chunks from vector DB
        retrieval_results = await chromadb_service.search_similar(
            query_embedding=query_embedding[0].tolist(),
            n_results=5
        )
        
        retrieval_time = time.time() - retrieval_start
        
        # Extract contexts from retrieval results
        contexts = []
        sources = []
        
        if retrieval_results and 'documents' in retrieval_results:
            documents = retrieval_results['documents'][0] if retrieval_results['documents'] else []
            metadatas = retrieval_results['metadatas'][0] if retrieval_results['metadatas'] else []
            distances = retrieval_results['distances'][0] if retrieval_results['distances'] else []
            
            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                contexts.append(doc)
                sources.append({
                    "document": metadata.get("filename", "unknown"),
                    "page": metadata.get("page", 1),
                    "chunk_id": metadata.get("chunk_id", f"chunk_{i}"),
                    "relevance_score": round(1 - distance, 3),  # Convert distance to similarity
                    "snippet": doc[:200] + "..." if len(doc) > 200 else doc
                })
        
        # 3. Generate response using LLM
        generation_start = time.time()
        
        if contexts:
            # Use RAG generation with retrieved contexts
            generation_result = await generation_service.generate_answer(user_query, contexts)
            response_text = generation_result["answer"]
        else:
            # Fallback to direct generation if no contexts found
            fallback_prompt = f"Answer this question based on your general knowledge: {user_query}"
            response_text = await generation_service.generate_simple(fallback_prompt)
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            sources=sources,
            metadata={
                "model_used": generation_service.model_name,
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": round(generation_time, 3),
                "total_time": round(total_time, 3),
                "total_chunks_retrieved": len(contexts),
                "embedding_model": embedding_service.model_name
            },
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        # Fallback to simple response on error
        logger.error(f"RAG pipeline error: {e}")
        return ChatResponse(
            response=f"I apologize, but I encountered an error processing your request: {str(e)}. Please try again or contact support if the issue persists.",
            session_id=session_id,
            sources=[],
            metadata={
                "error": str(e),
                "fallback_used": True
            },
            timestamp=datetime.utcnow()
        )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint."""
    
    async def generate_stream():
        """Generate streaming response."""
        session_id = request.session_id or str(uuid.uuid4())
        user_query = request.messages[-1].content if request.messages else ""
        
        # Mock streaming response
        response_text = f"This is a streaming mock response to: '{user_query}'. "
        response_text += "The actual RAG system will stream real-time responses from the language model."
        
        # Send metadata first
        metadata = {
            "type": "metadata",
            "session_id": session_id,
            "sources": [
                {
                    "document": "sample_document.pdf",
                    "page": 5,
                    "relevance_score": 0.85
                }
            ]
        }
        yield f"data: {json.dumps(metadata)}\n\n"
        
        # Stream response word by word
        words = response_text.split()
        for i, word in enumerate(words):
            chunk = {
                "type": "content",
                "content": word + " ",
                "is_final": i == len(words) - 1
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.05)  # Simulate typing delay
        
        # Send final signal
        final_chunk = {"type": "done"}
        yield f"data: {json.dumps(final_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )


@router.get("/sessions", response_model=List[SessionInfo])
async def list_sessions():
    """List all chat sessions."""
    
    # TODO: Implement actual session listing from database
    
    return [
        SessionInfo(
            session_id="session_1",
            title="Technical Documentation Query",
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            message_count=6
        ),
        SessionInfo(
            session_id="session_2",
            title="API Reference Questions",
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            message_count=4
        )
    ]


@router.get("/sessions/{session_id}", response_model=SessionHistory)
async def get_session_history(session_id: str):
    """Get complete history for a session."""
    
    # TODO: Implement actual session retrieval from database
    
    return SessionHistory(
        session_id=session_id,
        title="Sample Session",
        messages=[
            Message(role="user", content="What is the main purpose of this system?"),
            Message(role="assistant", content="This is a RAG system for querying technical documents."),
        ],
        created_at=datetime.utcnow(),
        last_updated=datetime.utcnow()
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    
    # TODO: Implement actual session deletion
    
    return {"message": f"Session {session_id} deleted successfully"} 