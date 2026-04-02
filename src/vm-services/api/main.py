from fastapi import FastAPI

app = FastAPI(title="RecSys Api", version = "1.0.0")



@app.get("/")
def read_root():
    return {"message": "Recommendation System API is running!", "status": "success"}

@app.get("/health")
def health_check():
    return {"message": "API is healthy!", "status": "healthy"}

@app.get("/recommendations")
def get_recommendations(user_id: int, top_n: int = 10):
    # Placeholder for recommendation logic
    # In a real implementation, you would query your recommendation engine here
    recommendations = [f"item_{i}" for i in range(1, top_n + 1)]
    
    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "status": "success"
    }


