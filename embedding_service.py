"""
Qwen embedding service for generating text embeddings.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional, Dict, Any
import logging
import gc
from contextlib import contextmanager

import sys
sys.path.append('./backend')
from core.config import settings
from core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class QwenEmbeddingService:
    """Service for generating embeddings using Qwen3-Embedding-8B model."""
    
    def __init__(self):
        """Initialize the embedding service."""
        self.model = None
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight for development
        self.embedding_dimension = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 256
        
    async def initialize(self):
        """Initialize the embedding model."""
        try:
            logger.info(f"Initializing embedding model on device: {self.device}")
            
            # Use lightweight model for development
            if settings.ENVIRONMENT == "development":
                self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.max_length = 256
            else:
                # Production: Use Qwen3-Embedding-8B
                self.model_name = "Qwen/Qwen3-Embedding-8B"
                self.max_length = 8192
            
            # Initialize model
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Get embedding dimension
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"✅ Embedding model initialized: {self.model_name}")
            logger.info(f"Embedding dimension: {self.embedding_dimension}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise EmbeddingError(f"Model initialization failed: {str(e)}")
    
    async def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for input texts."""
        if not self.model:
            raise EmbeddingError("Embedding model not initialized")
        
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            if not texts:
                raise EmbeddingError("No texts provided for embedding")
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=8,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingError(f"Embedding generation failed: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        if not self.embedding_dimension:
            raise EmbeddingError("Model not initialized")
        return self.embedding_dimension
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on embedding service."""
        try:
            if not self.model:
                return {"status": "unhealthy", "error": "Model not initialized"}
            
            # Test embedding generation
            test_text = "Health check test"
            embedding = self.model.encode([test_text])
            
            return {
                "status": "healthy",
                "model_name": self.model_name,
                "embedding_dimension": self.embedding_dimension,
                "device": self.device,
                "test_embedding_shape": embedding.shape
            }
            
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Global embedding service instance
embedding_service = QwenEmbeddingService() 