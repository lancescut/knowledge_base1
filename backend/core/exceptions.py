"""
Exception handlers for the RAG System API.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import traceback


class RAGException(Exception):
    """Base exception for RAG system."""
    
    def __init__(self, message: str, code: str = "RAG_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class DocumentProcessingError(RAGException):
    """Exception for document processing errors."""
    
    def __init__(self, message: str):
        super().__init__(message, "DOCUMENT_PROCESSING_ERROR")


class EmbeddingError(RAGException):
    """Exception for embedding generation errors."""
    
    def __init__(self, message: str):
        super().__init__(message, "EMBEDDING_ERROR")


class RetrievalError(RAGException):
    """Exception for retrieval errors."""
    
    def __init__(self, message: str):
        super().__init__(message, "RETRIEVAL_ERROR")


class GenerationError(RAGException):
    """Exception for text generation errors."""
    
    def __init__(self, message: str):
        super().__init__(message, "GENERATION_ERROR")


async def rag_exception_handler(request: Request, exc: RAGException):
    """Handle RAG system exceptions."""
    logger.error(f"RAG Exception: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.code,
            "message": exc.message,
            "detail": "An error occurred in the RAG system"
        }
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP_ERROR",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred",
            "detail": str(exc) if request.app.debug else "Internal server error"
        }
    )


def setup_exception_handlers(app: FastAPI):
    """Setup all exception handlers."""
    app.add_exception_handler(RAGException, rag_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler) 