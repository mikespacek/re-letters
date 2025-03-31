from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sys
import os
import json
from pathlib import Path
import uuid

# Import main.py functionality
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main

# Create a simplified app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001", 
        "http://localhost:3005", 
        "https://re-letters.vercel.app", 
        "https://*.vercel.app", 
        "*"
    ],  # Add specific origins including Vercel
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="."), name="static")

# Create temp directory
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)

# Create templates directory
TEMPLATES_DIR = Path("./templates")
TEMPLATES_DIR.mkdir(exist_ok=True)

# Create data directory
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# Simplified letter templates storage
LETTER_TEMPLATES = {}

# Load any existing templates
@app.on_event("startup")
async def startup_event():
    """Load templates on startup"""
    try:
        for template_file in TEMPLATES_DIR.glob("*.json"):
            try:
                with open(template_file, "r") as f:
                    template_data = json.load(f)
                    template_id = template_file.stem  # Use filename without extension as ID
                    LETTER_TEMPLATES[template_id] = template_data
                    print(f"Loaded template: {template_id}")
            except Exception as e:
                print(f"Error loading template {template_file}: {str(e)}")
    except Exception as e:
        print(f"Error on startup: {str(e)}")

# Serve HTML files
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the index.html file"""
    with open("index.html") as f:
        return f.read()

@app.get("/letters.html", response_class=HTMLResponse)
async def serve_letters():
    """Serve the letters.html file"""
    with open("letters.html") as f:
        return f.read()

@app.get("/test.html", response_class=HTMLResponse)
async def serve_test():
    """Serve the test.html file"""
    with open("test.html") as f:
        return f.read()

@app.get("/debug.html", response_class=HTMLResponse)
async def serve_debug():
    """Serve the debug.html file"""
    with open("debug.html") as f:
        return f.read()

@app.get("/api-test.html", response_class=HTMLResponse)
async def serve_api_test():
    """Serve the api-test.html file"""
    with open("api-test.html") as f:
        return f.read()

@app.get("/api")
async def api_root():
    """API Root endpoint"""
    return {
        "message": "RE Letters API is running",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/test",
            "/api-info",
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
        "api_directory": os.listdir("api") if os.path.exists("api") else "Not found",
        "templates_loaded": len(LETTER_TEMPLATES)
    }

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
    Upload a file for processing
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

# Proxy endpoints from main.py
@app.post("/process_excel_style")
async def process_excel_style_proxy(file: UploadFile = File(...), delimiter: str = Form(None)):
    """Proxy for the process_excel_style endpoint in main.py"""
    return await main.process_excel_style_endpoint(file, delimiter)

@app.post("/generate_letters")
async def generate_letters_proxy(request: Request):
    """Proxy for the generate_letters endpoint in main.py"""
    return await main.generate_letters(request)

@app.post("/print_letters")
async def print_letters_proxy(request: Request):
    """Proxy for the print_letters endpoint in main.py"""
    return await main.print_letters(request)

@app.post("/download")
async def download_proxy(file_name: str = Form(...), session_id: str = Form(...)):
    """Proxy for the download endpoint in main.py"""
    return await main.download(file_name, session_id)

# Add error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": str(type(exc).__name__)},
    )

# Export the handler for Vercel
handler = app 