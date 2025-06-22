"""
Task service for interfacing with Celery workers.
"""

from celery import Celery
from typing import Dict, Any, Optional
import logging

from core.config import settings

logger = logging.getLogger(__name__)


class TaskService:
    """Service for managing asynchronous tasks."""
    
    def __init__(self):
        """Initialize task service."""
        self.celery_app = None
        
    def initialize(self):
        """Initialize Celery client for task submission."""
        try:
            self.celery_app = Celery(
                broker=settings.REDIS_URL,
                backend=settings.REDIS_URL
            )
            logger.info("Task service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize task service: {e}")
            return False
    
    def submit_document_processing(self, job_id: str, file_path: str) -> Optional[str]:
        """
        Submit document processing task.
        
        Args:
            job_id: Unique job identifier
            file_path: Path to uploaded file
            
        Returns:
            Task ID if successful, None otherwise
        """
        if not self.celery_app:
            logger.error("Task service not initialized")
            return None
            
        try:
            # Submit task to Celery worker
            result = self.celery_app.send_task(
                'celery_worker.process_document',
                args=[job_id, file_path]
            )
            
            logger.info(f"Submitted document processing task: {result.id}")
            return result.id
            
        except Exception as e:
            logger.error(f"Failed to submit document processing task: {e}")
            return None
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get task status and result.
        
        Args:
            task_id: Celery task ID
            
        Returns:
            Task status information
        """
        if not self.celery_app:
            return {"status": "error", "message": "Task service not initialized"}
            
        try:
            result = self.celery_app.AsyncResult(task_id)
            
            return {
                "task_id": task_id,
                "status": result.status,
                "result": result.result if result.ready() else None,
                "successful": result.successful() if result.ready() else None,
                "failed": result.failed() if result.ready() else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return {"status": "error", "message": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on task service.
        
        Returns:
            Health status
        """
        if not self.celery_app:
            return {"status": "unhealthy", "error": "Not initialized"}
            
        try:
            # Submit health check task
            result = self.celery_app.send_task('celery_worker.health_check')
            
            # Wait for result with timeout
            task_result = result.get(timeout=5)
            
            return {
                "status": "healthy",
                "worker_response": task_result,
                "broker_url": settings.REDIS_URL
            }
            
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "broker_url": settings.REDIS_URL
            }


# Global task service instance
task_service = TaskService() 