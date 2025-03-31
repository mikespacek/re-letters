from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import tempfile
import os
import uuid
import shutil
from io import BytesIO
from pathlib import Path

app = FastAPI(title="CSV Processor API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a temporary directory for file storage
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)

def clean_data(df):
    """
    Clean and standardize CSV data
    """
    # Make a copy to avoid modifying the original
    clean_df = df.copy()
    
    # Process each column
    for col in clean_df.columns:
        # Convert to string if not already
        if clean_df[col].dtype != 'object':
            clean_df[col] = clean_df[col].astype(str)
            
        # Strip whitespace
        if clean_df[col].dtype == 'object':
            clean_df[col] = clean_df[col].str.strip()
            
        # Handle ZIP codes - ensure 5 digits with leading zeros
        if 'zip' in col.lower() or 'postal' in col.lower():
            clean_df[col] = clean_df[col].apply(
                lambda x: x.zfill(5) if x.isdigit() and len(x) <= 5 else x
            )
            
        # Handle phone numbers - standardize format
        if 'phone' in col.lower():
            clean_df[col] = clean_df[col].apply(
                lambda x: ''.join(c for c in x if c.isdigit())
            )
            
        # Remove dollar signs and commas from price/money fields
        if any(term in col.lower() for term in ['price', 'cost', 'value', 'sale']):
            clean_df[col] = clean_df[col].apply(
                lambda x: x.replace('$', '').replace(',', '') if isinstance(x, str) else x
            )
            
    return clean_df
    
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    output_format: str = Form(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload a CSV file, process it, and return in the requested format
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Create a unique filename for the processed file
        file_id = str(uuid.uuid4())
        input_path = TEMP_DIR / f"{file_id}_input.csv"
        
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Read CSV with pandas
        df = pd.read_csv(input_path)
        
        # Process the data
        cleaned_df = clean_data(df)
        
        # Prepare output file based on requested format
        output_filename = f"{file.filename.split('.')[0]}_processed"
        
        if output_format == "csv":
            output_path = TEMP_DIR / f"{file_id}_output.csv"
            cleaned_df.to_csv(output_path, index=False)
            media_type = "text/csv"
            download_filename = f"{output_filename}.csv"
            
        elif output_format == "excel":
            output_path = TEMP_DIR / f"{file_id}_output.xlsx"
            cleaned_df.to_excel(output_path, index=False)
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            download_filename = f"{output_filename}.xlsx"
            
        elif output_format == "numbers":
            # Since Numbers is Apple-specific, we'll create an Excel file that Numbers can open
            output_path = TEMP_DIR / f"{file_id}_output.xlsx"
            cleaned_df.to_excel(output_path, index=False)
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            download_filename = f"{output_filename}.xlsx"  # Numbers will convert this
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported output format")
        
        # Add cleanup task
        if background_tasks:
            background_tasks.add_task(cleanup_temp_file, [input_path, output_path])
        
        # Return the file
        return FileResponse(
            path=output_path,
            filename=download_filename,
            media_type=media_type,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def cleanup_temp_file(files):
    """Delete temporary files after processing"""
    for file in files:
        try:
            os.unlink(file)
        except:
            pass

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 