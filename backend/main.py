"""
Main FastAPI application for the Technical Knowledge Base RAG System.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from contextlib import asynccontextmanager
from loguru import logger

from core.config import settings
from api.v1.routes import chat, documents, health
from core.exceptions import setup_exception_handlers
from services.chromadb_service import chromadb_service
from services.task_service import task_service
from services.embedding_service import embedding_service
from services.generation_service import generation_service
from services.pdf_service import pdf_service
from services.chunking_service import chunking_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("🚀 Starting RAG System API...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"API Version: {settings.API_V1_STR}")
    
    # Initialize embedding service
    try:
        await embedding_service.initialize()
        logger.info("✅ Embedding service initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize embedding service: {e}")
        # Don't fail startup - continue with degraded functionality
    
    # Initialize generation service
    try:
        await generation_service.initialize()
        logger.info("✅ Generation service initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize generation service: {e}")
        # Don't fail startup - continue with degraded functionality
    
    # Initialize ChromaDB service
    try:
        await chromadb_service.initialize()
        logger.info("✅ ChromaDB service initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ChromaDB: {e}")
        # Don't fail startup - continue with degraded functionality
    
    # Initialize Task service
    try:
        task_service.initialize()
        logger.info("✅ Task service initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Task service: {e}")
        # Don't fail startup - continue with degraded functionality
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG System API...")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Technical Knowledge Base RAG System API",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup exception handlers
setup_exception_handlers(app)

# Include routers
app.include_router(health.router, prefix=f"{settings.API_V1_STR}/health", tags=["health"])
app.include_router(documents.router, prefix=f"{settings.API_V1_STR}/documents", tags=["documents"])
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["chat"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Technical Knowledge Base RAG System API",
        "version": "1.0.0",
        "status": "running",
        "docs": f"{settings.API_V1_STR}/docs"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.ENVIRONMENT == "development" else False,
        log_level="debug" if settings.ENVIRONMENT == "development" else "info"
    ) 