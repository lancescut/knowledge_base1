"""
ChromaDB service for vector database operations.
"""

import chromadb
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import logging

from core.config import settings
from core.exceptions import RetrievalError, EmbeddingError

logger = logging.getLogger(__name__)


class ChromaDBService:
    """ChromaDB vector database service."""
    
    def __init__(self):
        """Initialize ChromaDB client."""
        self.client = None
        self.collection = None
        self.collection_name = "rag_documents"
        
    async def initialize(self):
        """Initialize the ChromaDB client and collection."""
        try:
            # Use unauthenticated local ChromaDB client (auth providers removed in recent versions)
            self.client = chromadb.HttpClient(
                host=settings.CHROMADB_HOST,
                port=settings.CHROMADB_PORT,
                tenant="default",
                database="default"
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Connected to existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "description": "RAG system document chunks"
                    }
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
            logger.info("ChromaDB service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise RetrievalError(f"ChromaDB initialization failed: {str(e)}")
    
    async def add_documents(
        self, 
        documents: List[str], 
        embeddings: List[List[float]], 
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs
            
        Returns:
            bool: Success status
        """
        if not self.collection:
            raise RetrievalError("ChromaDB not initialized")
            
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
            
            # Add documents to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {str(e)}")
            raise RetrievalError(f"Failed to add documents: {str(e)}")
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Dict containing search results
        """
        if not self.collection:
            raise RetrievalError("ChromaDB not initialized")
            
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "ids": results["ids"][0] if results["ids"] else []
            }
            
            logger.info(f"Retrieved {len(formatted_results['documents'])} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {str(e)}")
            raise RetrievalError(f"Search failed: {str(e)}")
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.collection:
            raise RetrievalError("ChromaDB not initialized")
            
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            raise RetrievalError(f"Failed to get stats: {str(e)}")
    
    async def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        if not self.collection:
            raise RetrievalError("ChromaDB not initialized")
            
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise RetrievalError(f"Delete failed: {str(e)}")
    
    async def reset_collection(self) -> bool:
        """Reset the collection (delete all documents)."""
        if not self.client:
            raise RetrievalError("ChromaDB not initialized")
            
        try:
            # Delete existing collection
            if self.collection:
                self.client.delete_collection(name=self.collection_name)
            
            # Recreate collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "description": "RAG system document chunks"
                }
            )
            
            logger.info(f"Reset collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            raise RetrievalError(f"Reset failed: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            if not self.client:
                return {"status": "unhealthy", "error": "Client not initialized"}
                
            # Try to get collection info
            if self.collection:
                count = self.collection.count()
                return {
                    "status": "healthy",
                    "collection_name": self.collection_name,
                    "document_count": count,
                    "host": settings.CHROMADB_HOST,
                    "port": settings.CHROMADB_PORT
                }
            else:
                return {"status": "unhealthy", "error": "Collection not initialized"}
                
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Global ChromaDB service instance
chromadb_service = ChromaDBService() 