"""
Health check endpoints.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from datetime import datetime
import psutil
import os

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    environment: str
    uptime: float
    memory_usage: dict
    disk_usage: dict


class ComponentHealth(BaseModel):
    """Individual component health model."""
    name: str
    status: str
    details: dict = {}


class DetailedHealthResponse(BaseModel):
    """Detailed health check response."""
    status: str
    timestamp: datetime
    components: list[ComponentHealth]


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "development"),
        uptime=0.0,  # TODO: Implement actual uptime tracking
        memory_usage={
            "used": psutil.virtual_memory().used,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent
        },
        disk_usage={
            "used": psutil.disk_usage('/').used,
            "free": psutil.disk_usage('/').free,
            "percent": psutil.disk_usage('/').percent
        }
    )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """Detailed health check with component status."""
    components = []
    
    # Check ChromaDB
    try:
        # TODO: Implement actual ChromaDB health check
        components.append(ComponentHealth(
            name="chromadb",
            status="healthy",
            details={"host": os.getenv("CHROMADB_HOST", "localhost")}
        ))
    except Exception as e:
        components.append(ComponentHealth(
            name="chromadb",
            status="unhealthy",
            details={"error": str(e)}
        ))
    
    # Check Redis
    try:
        # TODO: Implement actual Redis health check
        components.append(ComponentHealth(
            name="redis",
            status="healthy",
            details={"url": os.getenv("REDIS_URL", "redis://localhost:6379/0")}
        ))
    except Exception as e:
        components.append(ComponentHealth(
            name="redis",
            status="unhealthy",
            details={"error": str(e)}
        ))
    
    # Overall status
    overall_status = "healthy" if all(c.status == "healthy" for c in components) else "unhealthy"
    
    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        components=components
    )


@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    # TODO: Implement actual readiness checks
    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {"status": "alive"} 