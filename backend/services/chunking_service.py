"""
Text chunking service for optimizing RAG retrieval.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic" 
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    HYBRID = "hybrid"


@dataclass
class ChunkConfig:
    """Configuration for chunking parameters."""
    max_chunk_size: int = 1000
    overlap_size: int = 200
    min_chunk_size: int = 100
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID


class TextChunkingService:
    """Service for intelligent text chunking optimized for RAG systems."""
    
    def __init__(self):
        """Initialize the chunking service."""
        self.default_config = ChunkConfig()
    
    def chunk_text(self, text: str, config: Optional[ChunkConfig] = None) -> List[Dict[str, Any]]:
        """Chunk text using the specified strategy."""
        if config is None:
            config = self.default_config
        
        try:
            # Use paragraph-based chunking as default
            chunks = self._split_by_paragraphs(text, config)
            
            # Convert to chunk objects with metadata
            result = []
            for i, chunk_text in enumerate(chunks):
                chunk = {
                    "id": f"chunk_{i}",
                    "text": chunk_text,
                    "metadata": {
                        "chunk_index": i,
                        "chunk_length": len(chunk_text),
                        "strategy": config.strategy.value,
                        "word_count": len(chunk_text.split())
                    }
                }
                result.append(chunk)
            
            logger.info(f"Created {len(result)} chunks using {config.strategy.value} strategy")
            return result
            
        except Exception as e:
            logger.error(f"Text chunking failed: {str(e)}")
            raise DocumentProcessingError(f"Chunking failed: {str(e)}")
    
    def _split_by_paragraphs(self, text: str, config: ChunkConfig) -> List[str]:
        """Split text by paragraphs, combining small ones."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If paragraph is too large, split it further
            if len(paragraph) > config.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) + 1 > config.max_chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                    else:
                        if temp_chunk:
                            temp_chunk += " " + sentence
                        else:
                            temp_chunk = sentence
                
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
            else:
                # Check if adding this paragraph would exceed max size
                if len(current_chunk) + len(paragraph) + 2 > config.max_chunk_size:
                    if current_chunk and len(current_chunk) >= config.min_chunk_size:
                        chunks.append(current_chunk.strip())
                        current_chunk = paragraph
                    else:
                        current_chunk = paragraph
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= config.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on chunking service."""
        try:
            test_text = "This is a test sentence. This is another test sentence."
            test_chunks = self.chunk_text(test_text)
            
            return {
                "status": "healthy",
                "available_strategies": [strategy.value for strategy in ChunkingStrategy],
                "test_result": {"chunks_created": len(test_chunks)}
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Global chunking service instance
chunking_service = TextChunkingService() 