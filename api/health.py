from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
@app.get("/api/health")
async def health_check():
    """Standalone health check endpoint"""
    return {
        "status": "ok",
        "message": "Health check endpoint is working"
    }

handler = app 