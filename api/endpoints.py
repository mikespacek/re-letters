from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from pathlib import Path
import uuid

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)

# Create templates directory
TEMPLATES_DIR = Path("./templates")
TEMPLATES_DIR.mkdir(exist_ok=True)

# Simplified letter templates storage
LETTER_TEMPLATES = {}

# Load any existing templates
@app.on_event("startup")
async def startup_event():
    try:
        for template_file in TEMPLATES_DIR.glob("*.json"):
            try:
                with open(template_file, "r") as f:
                    template_data = json.load(f)
                    template_id = template_file.stem  # Use filename without extension as ID
                    LETTER_TEMPLATES[template_id] = template_data
            except Exception as e:
                print(f"Error loading template {template_file}: {str(e)}")
    except Exception as e:
        print(f"Error on startup: {str(e)}")

@app.get("/templates")
async def get_templates():
    """Get all letter templates"""
    templates = []
    for template_id, template_data in LETTER_TEMPLATES.items():
        templates.append({
            "id": template_id,
            "name": template_data.get("name", "Unnamed"),
            "description": template_data.get("description", ""),
            "created": template_data.get("created", ""),
            "updated": template_data.get("updated", "")
        })
    return templates

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    delimiter: str = Form(",")
):
    """
    Simplified upload endpoint
    """
    try:
        # Create a unique session directory
        session_id = str(uuid.uuid4())
        session_dir = TEMP_DIR / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Save the file
        file_path = session_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "status": "success", 
            "message": "File uploaded successfully", 
            "session_id": session_id,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Make handler available for Vercel serverless
handler = app 