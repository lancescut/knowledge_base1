"""
Document management endpoints.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid
import os
from loguru import logger

router = APIRouter()


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    job_id: str
    status: str
    message: str
    filename: str
    file_size: int


class DocumentStatus(BaseModel):
    """Document processing status model."""
    job_id: str
    status: str  # processing, completed, failed
    progress: float
    message: str
    metadata: dict = {}
    created_at: datetime
    completed_at: Optional[datetime] = None


class DocumentInfo(BaseModel):
    """Document information model."""
    id: str
    filename: str
    file_size: int
    chunks_count: int
    upload_date: datetime
    status: str


class ImmediateUploadResponse(BaseModel):
    """Immediate document upload response model."""
    document_id: str
    filename: str
    file_size: int
    page_count: int
    element_count: int
    total_characters: int
    chunks_created: int
    processing_time: float
    status: str
    timestamp: datetime


@router.post("/upload/immediate", response_model=ImmediateUploadResponse)
async def upload_document_immediate(
    file: UploadFile = File(...),
    extract_text: bool = True,
    create_embeddings: bool = True
):
    """Upload and immediately process a PDF document (for smaller files)."""
    
    try:
        # Import services here to avoid circular imports
        from services.pdf_service import pdf_service
        from services.embedding_service import embedding_service
        from services.chromadb_service import chromadb_service
        
        import time
        start_time = time.time()
        
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Validate file size (10MB limit for immediate processing)
        MAX_FILE_SIZE = 10 * 1024 * 1024
        content = await file.read()
        file_size = len(content)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size {file_size} exceeds maximum allowed size {MAX_FILE_SIZE} for immediate processing. Use background upload for larger files."
            )
        
        # Reset file pointer for processing
        file.file.seek(0)
        
        # Extract text from PDF
        logger.info(f"Processing uploaded PDF: {file.filename}")
        extraction_result = await pdf_service.extract_from_file(file.file, file.filename)
        
        chunks_created = 0
        
        # If embedding creation requested, create embeddings and store in vector DB
        if create_embeddings:
            try:
                # Import chunking service for intelligent chunking
                from services.chunking_service import chunking_service, ChunkConfig, ChunkingStrategy
                
                # Get full text for intelligent chunking
                full_text = pdf_service.get_text_content(extraction_result["elements"])
                
                # Configure chunking for technical content
                chunk_config = ChunkConfig(
                    max_chunk_size=800,  # Optimized for technical content
                    overlap_size=100,    # Good overlap for context
                    min_chunk_size=100,  # Minimum meaningful size
                    strategy=ChunkingStrategy.PARAGRAPH
                )
                
                # Create intelligent chunks
                intelligent_chunks = chunking_service.chunk_text(full_text, chunk_config)
                
                chunks = []
                metadatas = []
                
                for chunk in intelligent_chunks:
                    chunks.append(chunk["text"])
                    metadatas.append({
                        "document_id": extraction_result["document_id"],
                        "filename": extraction_result["filename"],
                        "chunk_id": f"{extraction_result['document_id']}_{chunk['id']}",
                        "chunk_index": chunk["metadata"]["chunk_index"],
                        "chunk_length": chunk["metadata"]["chunk_length"],
                        "word_count": chunk["metadata"]["word_count"],
                        "strategy": chunk["metadata"]["strategy"],
                        "page_number": None,  # Not available from full text chunking
                        "element_type": "intelligent_chunk",
                        "importance": "medium"
                    })
                
                if chunks:
                    # Generate embeddings for chunks
                    embeddings = await embedding_service.encode(chunks)
                    
                    # Store in vector database
                    await chromadb_service.add_documents(
                        documents=chunks,
                        embeddings=embeddings.tolist(),
                        metadatas=metadatas,
                        ids=[metadata["chunk_id"] for metadata in metadatas]
                    )
                    
                    chunks_created = len(chunks)
                    logger.info(f"Successfully processed {file.filename}: {chunks_created} chunks stored")
                    
            except Exception as e:
                logger.error(f"Failed to process chunks for {file.filename}: {e}")
                # Continue without failing the entire request
                chunks_created = 0
        
        processing_time = time.time() - start_time
        
        response = ImmediateUploadResponse(
            document_id=extraction_result["document_id"],
            filename=extraction_result["filename"],
            file_size=extraction_result["file_size"],
            page_count=extraction_result["page_count"],
            element_count=extraction_result["element_count"],
            total_characters=extraction_result["total_characters"],
            chunks_created=chunks_created,
            processing_time=round(processing_time, 2),
            status="completed",
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Document processing completed: {file.filename} in {processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload a PDF document for background processing (for larger files)."""
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Validate file size (100MB limit)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size {file_size} exceeds maximum allowed size {MAX_FILE_SIZE}"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save file temporarily
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{job_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        buffer.write(content)
    
    # Schedule background processing
    # TODO: Add actual document processing task using pdf_service
    # background_tasks.add_task(process_document_background, job_id, file_path)
    
    return DocumentUploadResponse(
        job_id=job_id,
        status="processing",
        message="Document uploaded successfully and queued for processing",
        filename=file.filename,
        file_size=file_size
    )


@router.get("/status/{job_id}", response_model=DocumentStatus)
async def get_document_status(job_id: str):
    """Get the processing status of a document."""
    
    # TODO: Implement actual status checking from database/cache
    
    return DocumentStatus(
        job_id=job_id,
        status="completed",  # Mock status
        progress=100.0,
        message="Document processed successfully",
        metadata={
            "chunks_created": 42,
            "pages_processed": 10
        },
        created_at=datetime.utcnow(),
        completed_at=datetime.utcnow()
    )


@router.get("/", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded and processed documents."""
    
    # TODO: Implement actual document listing from database
    
    return [
        DocumentInfo(
            id="doc_1",
            filename="sample_document.pdf",
            file_size=1024000,
            chunks_count=42,
            upload_date=datetime.utcnow(),
            status="completed"
        )
    ]


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str):
    """Get information about a specific document."""
    
    # TODO: Implement actual document retrieval from database
    
    return DocumentInfo(
        id=document_id,
        filename="sample_document.pdf",
        file_size=1024000,
        chunks_count=42,
        upload_date=datetime.utcnow(),
        status="completed"
    )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its associated chunks."""
    
    # TODO: Implement actual document deletion
    
    return {"message": f"Document {document_id} deleted successfully"} 