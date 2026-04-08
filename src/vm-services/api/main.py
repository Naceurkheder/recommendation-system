"""
High-Performance Recommendation Engine API
Exposes optimized C implementation via REST endpoints
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import logging

from rec_engine_wrapper import get_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RecSys API",
    version="2.0.0",
    description="High-Performance Recommendation Engine - OpenMP/MPI/CUDA"
)

# Request/Response models
class SimilarUsersRequest(BaseModel):
    user_id: int
    k: int = 10

class SimilarUsersResponse(BaseModel):
    user_id: int
    similar_users: List[int]
    status: str

class ItemRecommendationsRequest(BaseModel):
    user_id: int
    k: int = 10
    num_neighbors: int = 10

class ItemRecommendationsResponse(BaseModel):
    user_id: int
    recommendations: List[int]
    status: str

class SimilarityScoreResponse(BaseModel):
    user_a: int
    user_b: int
    similarity: float
    status: str

class EngineStatusResponse(BaseModel):
    status: str
    num_users: int
    num_items: int
    initialized: bool

# Global engine instance
_engine = None

def init_engine():
    """Initialize the recommendation engine"""
    global _engine
    try:
        _engine = get_engine()
        
        # Find CSV data file
        csv_path = os.getenv(
            'REC_ENGINE_DATA',
            '/vagrant/src/host-cuda/openmp/data/matrix.csv'
        )
        
        if not os.path.exists(csv_path):
            # Try alternate paths
            alternate_paths = [
                './data/matrix.csv',
                '../data/matrix.csv',
                '../../data/matrix.csv',
                '/home/data/matrix.csv',
            ]
            for alt_path in alternate_paths:
                if os.path.exists(alt_path):
                    csv_path = alt_path
                    break
        
        logger.info(f"Initializing recommendation engine with data: {csv_path}")
        
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return False
        
        success = _engine.init(csv_path)
        if success:
            num_users, num_items = _engine.get_dimensions()
            logger.info(
                f"✓ Engine initialized: {num_users} users, {num_items} items"
            )
            _engine.print_status()
        else:
            logger.error("Failed to initialize engine")
        
        return success
    except Exception as e:
        logger.error(f"Error initializing engine: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize engine on startup"""
    if not init_engine():
        logger.warning("Recommendation engine initialization failed - API will return errors")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global _engine
    if _engine:
        _engine.cleanup()
        logger.info("Engine cleaned up")

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Recommendation System API is running!",
        "version": "2.0.0",
        "status": "success"
    }

@app.get("/health")
def health_check() -> EngineStatusResponse:
    """Health check endpoint"""
    if not _engine:
        return EngineStatusResponse(
            status="unhealthy",
            num_users=0,
            num_items=0,
            initialized=False
        )
    
    try:
        num_users, num_items = _engine.get_dimensions()
        return EngineStatusResponse(
            status="healthy",
            num_users=num_users,
            num_items=num_items,
            initialized=True
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return EngineStatusResponse(
            status="error",
            num_users=0,
            num_items=0,
            initialized=False
        )

@app.post("/recommendations/similar-users")
def get_similar_users(request: SimilarUsersRequest) -> SimilarUsersResponse:
    """
    Get k most similar users for a given user
    
    Query similar users based on cosine similarity of rating vectors.
    Useful for user-based collaborative filtering.
    """
    if not _engine:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine not initialized"
        )
    
    try:
        similar_users = _engine.get_similar_users(
            request.user_id, request.k
        )
        return SimilarUsersResponse(
            user_id=request.user_id,
            similar_users=similar_users,
            status="success"
        )
    except Exception as e:
        logger.error(f"Error getting similar users: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommendations/items")
def get_item_recommendations(
    request: ItemRecommendationsRequest
) -> ItemRecommendationsResponse:
    """
    Get top-k item recommendations for a user
    
    Uses collaborative filtering with similar users to predict ratings
    for items the user hasn't rated yet.
    """
    if not _engine:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine not initialized"
        )
    
    try:
        recommendations = _engine.get_item_recommendations(
            request.user_id,
            request.k,
            request.num_neighbors
        )
        return ItemRecommendationsResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            status="success"
        )
    except Exception as e:
        logger.error(f"Error getting item recommendations: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/recommendations/similarity")
def get_similarity(
    user_a: int = Query(..., description="First user ID"),
    user_b: int = Query(..., description="Second user ID")
) -> SimilarityScoreResponse:
    """
    Get cosine similarity between two users
    
    Returns the cosine similarity score in range [-1, 1]
    """
    if not _engine:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine not initialized"
        )
    
    try:
        similarity = _engine.get_similarity(user_a, user_b)
        return SimilarityScoreResponse(
            user_a=user_a,
            user_b=user_b,
            similarity=similarity,
            status="success"
        )
    except Exception as e:
        logger.error(f"Error getting similarity: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/recommendations/top-k")
def get_recommendations(
    user_id: int = Query(..., description="User ID"),
    top_n: int = Query(10, ge=1, le=100, description="Number of recommendations")
):
    """
    Quick endpoint to get top-n item recommendations
    """
    return get_item_recommendations(
        ItemRecommendationsRequest(user_id=user_id, k=top_n)
    )

@app.get("/status")
def get_status() -> dict:
    """Get engine status and information"""
    if not _engine:
        return {"status": "not_initialized"}
    
    try:
        num_users, num_items = _engine.get_dimensions()
        return {
            "status": "initialized",
            "num_users": num_users,
            "num_items": num_items,
            "features": [
                "OpenMP-optimized similarity computation",
                "Cosine similarity metrics",
                "Collaborative filtering recommendations",
                "Top-k efficient selection"
            ]
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


