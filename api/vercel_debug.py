from fastapi import FastAPI
import sys
import os

app = FastAPI()

@app.get("/api/debug")
async def debug():
    """Debug endpoint to check environment and imports"""
    return {
        "status": "ok",
        "python_version": sys.version,
        "environment_variables": dict(os.environ),
        "working_directory": os.getcwd(),
        "directory_contents": os.listdir(),
        "api_directory_contents": os.listdir("api") if os.path.exists("api") else "Not found",
        "message": "Debug endpoint is working"
    }

handler = app 