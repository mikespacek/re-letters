from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import sys
import os

app = FastAPI()

@app.get("/api/minimal")
@app.get("/minimal")
async def minimal():
    """Minimal endpoint with no external imports"""
    return {
        "status": "ok",
        "message": "Minimal API is working",
        "python_version": sys.version
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Minimal health check is working"
    }

@app.get("/test")
async def test_endpoint():
    """Test endpoint"""
    return {
        "status": "ok",
        "message": "Minimal test endpoint is working"
    }

# Add error handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": str(type(exc).__name__)},
    )

handler = app 