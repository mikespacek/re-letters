from fastapi import FastAPI

app = FastAPI()

@app.get("/api/minimal")
async def minimal():
    """Minimal endpoint with no external imports"""
    return {
        "status": "ok",
        "message": "Minimal API is working"
    }

handler = app 