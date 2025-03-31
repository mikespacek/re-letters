from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Create a simplified app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RE Letters API is running",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/test",
            "/templates",
            "/upload",
            "/download"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "API is healthy"}

@app.get("/test")
async def test_endpoint():
    """Test endpoint"""
    return {"status": "ok", "message": "Test endpoint is working"}

@app.get("/api-info")
async def api_info():
    """Returns information about the API environment"""
    return {
        "python_version": sys.version,
        "environment": {k: v for k, v in os.environ.items() if not k.startswith("AWS_")},
        "directory": os.getcwd(),
        "files": os.listdir(),
        "api_directory": os.listdir("api") if os.path.exists("api") else "Not found"
    }

# Add error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": str(type(exc).__name__)},
    )

# Export the handler for Vercel
handler = app 