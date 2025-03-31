from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse
import pandas as pd
import tempfile
import os
import uuid
import shutil
import logging
import traceback
import base64
from io import BytesIO
from pathlib import Path
import csv
from collections import Counter
import json
from typing import Dict, List, Union, Any, Optional
from pydantic import BaseModel

# Set up logging - increase level for more detailed logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define API models
class TemplateBase(BaseModel):
    name: str
    description: Optional[str] = None
    content: str

class TemplateCreate(TemplateBase):
    pass

class TemplateUpdate(TemplateBase):
    pass

class TemplateResponse(TemplateBase):
    id: str
    created: str
    updated: str

class LetterContent(BaseModel):
    content: str
    variables: Optional[Dict[str, str]] = None

class LettersRequest(BaseModel):
    letters: List[LetterContent]
    output_format: str = "html"

app = FastAPI(title="CSV Processor API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", 
                   "http://localhost:3003", "http://localhost:3005", "http://127.0.0.1:3001", 
                   "https://*.vercel.app", "https://*.now.sh", "*"],  # Added Vercel domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a temporary directory for file storage
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)

# Create a directory for letter templates
TEMPLATES_DIR = Path("./templates")
TEMPLATES_DIR.mkdir(exist_ok=True)

# Store letter templates in memory for quick access
LETTER_TEMPLATES = {}

# Load existing templates if they exist
def load_templates():
    """Load letter templates from the templates directory"""
    global LETTER_TEMPLATES
    
    logger.info("Loading letter templates from disk")
    
    if not TEMPLATES_DIR.exists():
        logger.warning("Templates directory doesn't exist, creating it")
        TEMPLATES_DIR.mkdir(exist_ok=True)
    
    templates_found = False
    for template_file in TEMPLATES_DIR.glob("*.json"):
        try:
            with open(template_file, "r") as f:
                template_data = json.load(f)
                template_id = template_file.stem  # Use filename without extension as ID
                LETTER_TEMPLATES[template_id] = template_data
                logger.info(f"Loaded template: {template_id} - {template_data.get('name', 'Unnamed')}")
                templates_found = True
        except Exception as e:
            logger.error(f"Error loading template {template_file}: {str(e)}")
    
    # If no templates found, check for a default template
    if not templates_found:
        load_default_template()

def load_default_template():
    """Load the default template if it exists, or create one if no templates exist"""
    logger.info("No templates found, checking for default template")
    
    default_path = Path("./default_template.json")
    if default_path.exists():
        try:
            with open(default_path, "r") as f:
                template_data = json.load(f)
                template_id = str(uuid.uuid4())
                
                # Add timestamps
                import datetime
                now = datetime.datetime.now().isoformat()
                template_data["created"] = now
                template_data["updated"] = now
                
                # Save to memory and disk
                LETTER_TEMPLATES[template_id] = template_data
                
                # Save to templates directory
                template_path = TEMPLATES_DIR / f"{template_id}.json"
                with open(template_path, "w") as f:
                    json.dump(template_data, f, indent=2)
                
                logger.info(f"Created default template: {template_id} - {template_data.get('name', 'Default Template')}")
        except Exception as e:
            logger.error(f"Error loading default template: {str(e)}")
    else:
        logger.warning("No default template found")

# Load templates on startup
load_templates()

# Template variable patterns - these are the variables users can include in their templates
TEMPLATE_VARIABLES = {
    # Property owner variables
    "{first_name}": "Owner's first name",
    "{last_name}": "Owner's last name", 
    "{full_name}": "Owner's full name (First Last)",
    "{address}": "Property address",
    "{city}": "Property city",
    "{state}": "Property state",
    "{zip}": "Property ZIP code",
    
    # Special variables for different recipient types
    "{current_renter}": "Will be replaced with 'Current Renter' for renter properties",
    "{current_owner}": "Will be replaced with 'Current Owner' for investor properties with empty names",
    
    # Date variables
    "{current_date}": "Today's date (MM/DD/YYYY)",
    "{current_month}": "Current month name",
    "{current_year}": "Current year"
}

def clean_data(df):
    """
    Clean and standardize CSV data
    """
    logger.debug(f"Starting data cleaning. DataFrame shape: {df.shape}")
    # Make a copy to avoid modifying the original
    clean_df = df.copy()
    
    # Log columns for debugging
    logger.debug(f"Columns in DataFrame: {clean_df.columns.tolist()}")
    
    # Process each column
    for col in clean_df.columns:
        logger.debug(f"Processing column: {col}")
        # Convert to string if not already
        if clean_df[col].dtype != 'object':
            logger.debug(f"Converting column {col} to string")
            clean_df[col] = clean_df[col].astype(str)
            
        # Strip whitespace
        if clean_df[col].dtype == 'object':
            logger.debug(f"Stripping whitespace from column {col}")
            clean_df[col] = clean_df[col].str.strip()
            
        # Handle ZIP codes - ensure 5 digits with leading zeros
        if 'zip' in col.lower() or 'postal' in col.lower():
            logger.debug(f"Processing ZIP code column {col}")
            clean_df[col] = clean_df[col].apply(
                lambda x: x.zfill(5) if isinstance(x, str) and x.isdigit() and len(x) <= 5 else x
            )
            
        # Handle phone numbers - standardize format
        if 'phone' in col.lower():
            logger.debug(f"Processing phone number column {col}")
            clean_df[col] = clean_df[col].apply(
                lambda x: ''.join(c for c in x if isinstance(x, str) and c.isdigit()) if isinstance(x, str) else x
            )
            
        # Remove dollar signs and commas from price/money fields
        if any(term in col.lower() for term in ['price', 'cost', 'value', 'sale']):
            logger.debug(f"Processing price column {col}")
            clean_df[col] = clean_df[col].apply(
                lambda x: x.replace('$', '').replace(',', '') if isinstance(x, str) else x
            )
    
    logger.debug(f"Data cleaning complete. DataFrame shape: {clean_df.shape}")
    return clean_df

def fill_empty_names(df):
    """
    Fill empty first and last name fields using mail owner name
    """
    logger.debug("Filling empty first and last names from mail owner name")
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Find column names for mail owner, first name, and last name
    mail_owner_col = None
    first_name_col = None
    last_name_col = None
    
    for col in processed_df.columns:
        col_lower = col.lower()
        if 'mail owner' in col_lower or 'mail_owner' in col_lower:
            mail_owner_col = col
            logger.debug(f"Found mail owner column: {col}")
        elif 'first name' in col_lower or 'firstname' in col_lower:
            first_name_col = col
            logger.debug(f"Found first name column: {col}")
        elif 'last name' in col_lower or 'lastname' in col_lower:
            last_name_col = col
            logger.debug(f"Found last name column: {col}")
    
    # If we don't have all the required columns, return the original DataFrame
    if not all([mail_owner_col, first_name_col, last_name_col]):
        logger.warning("Could not find all required name columns")
        logger.debug(f"Available columns: {processed_df.columns.tolist()}")
        return processed_df
    
    # Count empty first/last names before filling
    empty_first_names = processed_df[first_name_col].isna().sum() + (processed_df[first_name_col] == '').sum()
    empty_last_names = processed_df[last_name_col].isna().sum() + (processed_df[last_name_col] == '').sum()
    logger.debug(f"Empty first names before processing: {empty_first_names}")
    logger.debug(f"Empty last names before processing: {empty_last_names}")
    
    # Process each row where first or last name is empty
    for idx, row in processed_df.iterrows():
        # Check if we have a mail owner name to use
        mail_owner = str(row[mail_owner_col]).strip()
        if not mail_owner or mail_owner.lower() == 'nan':
            continue
            
        # Check if first name is empty
        first_name = str(row[first_name_col]).strip()
        if not first_name or first_name.lower() == 'nan':
            # Process mail owner name to extract first name
            name_parts = mail_owner.split()
            
            if len(name_parts) >= 1:
                # Assign first part as first name
                processed_df.at[idx, first_name_col] = name_parts[0]
                logger.debug(f"Row {idx}: Filled first name with '{name_parts[0]}' from '{mail_owner}'")
        
        # Check if last name is empty
        last_name = str(row[last_name_col]).strip()
        if not last_name or last_name.lower() == 'nan':
            # Process mail owner name to extract last name
            name_parts = mail_owner.split()
            
            if len(name_parts) > 1:
                # Assign last part as last name
                processed_df.at[idx, last_name_col] = name_parts[-1]
                logger.debug(f"Row {idx}: Filled last name with '{name_parts[-1]}' from '{mail_owner}'")
            elif len(name_parts) == 1:
                # If only one name part and it's not already used for first name, use it for last name
                if first_name and first_name != name_parts[0]:
                    processed_df.at[idx, last_name_col] = name_parts[0]
                    logger.debug(f"Row {idx}: Used single name part '{name_parts[0]}' as last name")
    
    # Count remaining empty names after filling
    empty_first_names_after = processed_df[first_name_col].isna().sum() + (processed_df[first_name_col] == '').sum()
    empty_last_names_after = processed_df[last_name_col].isna().sum() + (processed_df[last_name_col] == '').sum()
    logger.debug(f"Empty first names after processing: {empty_first_names_after}")
    logger.debug(f"Empty last names after processing: {empty_last_names_after}")
    logger.debug(f"Filled {empty_first_names - empty_first_names_after} first names and {empty_last_names - empty_last_names_after} last names")
    
    return processed_df

def detect_delimiter(file_path, sample_lines=10):
    """
    Auto-detect the delimiter used in a CSV file by analyzing common delimiters
    """
    common_delimiters = [',', '\t', ';', '|']
    with open(file_path, 'r', newline='', errors='replace') as f:
        # Read a sample of lines
        sample = []
        for _ in range(sample_lines):
            line = f.readline()
            if not line:
                break
            sample.append(line)
        
        if not sample:
            logger.warning("Empty file, cannot detect delimiter")
            return ','  # Default to comma
        
        # Count occurrences of each delimiter in the sample
        counts = {}
        for delimiter in common_delimiters:
            counts[delimiter] = sum(line.count(delimiter) for line in sample) / len(sample)
        
        logger.debug(f"Delimiter counts: {counts}")
        
        # Return the delimiter with highest average count
        most_common = max(counts.items(), key=lambda x: x[1])
        if most_common[1] == 0:
            logger.warning("Could not detect any common delimiter, defaulting to comma")
            return ','
        
        logger.info(f"Detected delimiter: '{most_common[0]}' with average count {most_common[1]}")
        return most_common[0]

@app.post("/upload", response_class=Response)
async def upload_file(
    file: UploadFile = File(...),
    output_format: str = Form(...),
    filter_type: str = Form("all"),
    delimiter: str = Form(','),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload a CSV file, process it, and return in the requested format
    """
    logger.info(f"Starting file upload: {file.filename}, format: {output_format}, filter: {filter_type}, delimiter: {delimiter}")
    
    if not file.filename.endswith('.csv'):
        logger.error(f"Invalid file type: {file.filename}")
        return JSONResponse(
            status_code=400,
            content={"detail": "Only CSV files are supported"}
        )
    
    try:
        # Create a unique filename for the processed file
        file_id = str(uuid.uuid4())
        input_path = TEMP_DIR / f"{file_id}_input.csv"
        
        logger.info(f"Saving uploaded file to {input_path}")
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            content = await file.read()
            logger.debug(f"Read {len(content)} bytes from uploaded file")
            buffer.write(content)
        
        # Reset file pointer for future operations
        await file.seek(0)
        
        # Check if file was saved correctly
        if not input_path.exists():
            logger.error(f"Failed to save file to {input_path}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Failed to save uploaded file"}
            )
            
        file_size = input_path.stat().st_size
        logger.debug(f"Saved file size: {file_size} bytes")
        
        if file_size == 0:
            logger.error("Uploaded file is empty")
            return JSONResponse(
                status_code=400,
                content={"detail": "Uploaded file is empty"}
            )
        
        # Auto-detect delimiter if needed, overriding the provided one
        if delimiter == 'auto':
            detected_delimiter = detect_delimiter(input_path)
            logger.info(f"Auto-detected delimiter: '{detected_delimiter}'")
            delimiter = detected_delimiter
        
        # Handle special case for tab
        if delimiter == '\\t':
            delimiter = '\t'
        
        # Read CSV with pandas - using provided or detected delimiter
        logger.info(f"Reading CSV with pandas using delimiter: '{delimiter}'")
        try:
            logger.debug(f"Using delimiter: '{delimiter}'")
            
            # Read with provided delimiter
            df = pd.read_csv(input_path, delimiter=delimiter, dtype=str, on_bad_lines='warn')
            logger.debug(f"Read CSV successfully. Shape: {df.shape}, Columns: {df.columns.tolist()}")
            
            # If we only got one column, it might be a malformed CSV, try another approach
            if len(df.columns) <= 1:
                logger.warning("CSV appears malformed with only one column, attempting to detect correct delimiter")
                # Try to detect the correct delimiter
                best_delimiter = detect_delimiter(input_path)
                if best_delimiter != delimiter:
                    logger.info(f"Detected better delimiter: '{best_delimiter}', trying again")
                    df = pd.read_csv(input_path, delimiter=best_delimiter, dtype=str, on_bad_lines='warn')
                    logger.debug(f"Re-read CSV with new delimiter. Shape: {df.shape}, Columns: {df.columns.tolist()}")
                
                # If still malformed, try the CSV module approach
                if len(df.columns) <= 1:
                    logger.debug("Still malformed, attempting to fix by reading with Python's csv module")
                    import csv
                    from io import StringIO
                    
                    rows = []
                    with open(input_path, 'r', newline='', errors='replace') as f:
                        # Try with the detected delimiter first
                        reader = csv.reader(f, delimiter=best_delimiter)
                        for row in reader:
                            rows.append(row)
                    
                    if rows:
                        # Get max columns from any row
                        max_cols = max(len(row) for row in rows)
                        logger.debug(f"Found {max_cols} columns in CSV")
                        
                        # Pad rows with empty values if needed
                        padded_rows = []
                        for row in rows:
                            padded_rows.append(row + [''] * (max_cols - len(row)))
                        
                        # Convert to DataFrame
                        header = padded_rows[0]
                        data = padded_rows[1:]
                        df = pd.DataFrame(data, columns=header)
                        logger.debug(f"Created DataFrame from raw CSV. Shape: {df.shape}")
            
        except Exception as e:
            logger.error(f"Error reading CSV: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=400,
                content={"detail": f"Error reading CSV: {str(e)}"}
            )
        
        # Process the data
        logger.info("Cleaning and processing data")
        try:
            cleaned_df = clean_data(df)
            
            # Fill empty first and last names from mail owner name
            logger.info("Filling empty names from mail owner name")
            cleaned_df = fill_empty_names(cleaned_df)
            
            if filter_type != "all":
                # Apply filter based on filter_type
                logger.info(f"Applying filter: {filter_type}")
                
                # Check if "Owner Occupied" column exists
                owner_occupied_col = None
                for col in cleaned_df.columns:
                    if "owner occupied" in col.lower():
                        owner_occupied_col = col
                        logger.debug(f"Found Owner Occupied column: {col}")
                        break
                
                if owner_occupied_col:
                    # Log unique values in this column to debug
                    unique_values = cleaned_df[owner_occupied_col].unique()
                    logger.debug(f"Unique values in {owner_occupied_col}: {unique_values}")
                    
                    # Add specific logging for each unique value to help with debugging
                    for val in unique_values:
                        val_count = (cleaned_df[owner_occupied_col] == val).sum()
                        logger.debug(f"Value '{val}' appears {val_count} times")
                    
                    # Check for owner name columns
                    owner_name_cols = []
                    for col in cleaned_df.columns:
                        if any(term in col.lower() for term in ["owner first name", "owner last name", "mail owner name"]):
                            owner_name_cols.append(col)
                            logger.debug(f"Found owner name column: {col}")
                    
                    if filter_type == "owner_occupied":
                        # Owner occupied = yes filtering logic
                        logger.debug("Filtering for owner occupied properties")
                        # [... existing owner_occupied filtering logic ...]
                        has_positive_values = any(val.upper().strip() in ["Y", "YES", "TRUE", "T", "1", "OWNER", "OCCUPIED"] 
                                              for val in cleaned_df[owner_occupied_col].fillna('') if isinstance(val, str))
                        logger.debug(f"Has positive owner occupied values: {has_positive_values}")
                        
                        owner_occupied_mask = cleaned_df[owner_occupied_col].str.upper().fillna('').str.strip().isin(["Y", "YES", "TRUE", "T", "1", "OWNER", "OCCUPIED"])
                        # If no records match our positive value check, assume field has different format
                        if not owner_occupied_mask.any() and not has_positive_values:
                            logger.warning("No records match positive values in Owner Occupied column")
                            # Try a different approach - look for non-empty values in owner name columns
                            if owner_name_cols:
                                logger.debug("Trying alternative approach based on owner name values")
                                # If we have owner name fields populated, this could indicate owner occupied
                                owner_occupied_mask = cleaned_df[owner_name_cols[0]].notna() & (cleaned_df[owner_name_cols[0]] != '')
                        
                        filtered_df = cleaned_df[owner_occupied_mask]
                        logger.debug(f"Owner occupied filter matched {len(filtered_df)} of {len(cleaned_df)} rows")
                        
                    elif filter_type == "renter":
                        # Renter filtering logic
                        logger.debug("Filtering for renter properties")
                        # [... existing renter filtering logic ...]
                        has_negative_values = any(val.upper().strip() in ["N", "NO", "FALSE", "F", "0", "RENTER"] 
                                             for val in cleaned_df[owner_occupied_col].fillna('') if isinstance(val, str))
                        logger.debug(f"Has negative owner occupied values: {has_negative_values}")
                        
                        renter_mask = cleaned_df[owner_occupied_col].str.upper().fillna('').str.strip().isin(["N", "NO", "FALSE", "F", "0", "RENTER"])
                        
                        # If no records match our negative value check, assume empty means not owner occupied
                        if not renter_mask.any() and not has_negative_values:
                            logger.warning("No records match negative values in Owner Occupied column")
                            renter_mask = cleaned_df[owner_occupied_col].isna() | (cleaned_df[owner_occupied_col] == '')
                        
                        filtered_df = cleaned_df[renter_mask].copy()
                        
                        # Replace owner name fields with "Current Renter"
                        for col in owner_name_cols:
                            logger.debug(f"Replacing values in {col} with 'Current Renter'")
                            filtered_df[col] = "Current Renter"
                            
                        logger.debug(f"Renter filter matched {len(filtered_df)} of {len(cleaned_df)} rows")
                        
                    elif filter_type == "investor":
                        # Investor filtering logic
                        logger.debug("Filtering for investor properties")
                        # [... existing investor filtering logic ...]
                        investor_mask = cleaned_df[owner_occupied_col].str.upper().fillna('').str.strip().isin(["N", "NO", "FALSE", "F", "0", "RENTER"])
                        
                        # If no explicit negative values, consider empty values as not owner occupied
                        if not investor_mask.any():
                            logger.warning("No explicit negative values found, checking for empty/NA values")
                            investor_mask = cleaned_df[owner_occupied_col].isna() | (cleaned_df[owner_occupied_col] == '')
                            
                        # Additional check: empty values only count as investor if we have owner name data
                        if owner_name_cols:
                            has_owner_data = False
                            for col in owner_name_cols:
                                if cleaned_df[col].notna().any():
                                    has_owner_data = True
                                    break
                            
                            if has_owner_data:
                                logger.debug("Using owner name data to determine investor properties")
                                non_empty_owner = cleaned_df[owner_name_cols[0]].notna() & (cleaned_df[owner_name_cols[0]] != '')
                                investor_mask = investor_mask & non_empty_owner
                        
                        filtered_df = cleaned_df[investor_mask].copy()
                        
                        # For investor properties, replace empty first and last names with "Current Owner"
                        first_name_col = None
                        last_name_col = None
                        
                        for col in filtered_df.columns:
                            col_lower = col.lower()
                            if 'first name' in col_lower:
                                first_name_col = col
                                logger.debug(f"Found first name column for investor replacement: {col}")
                            elif 'last name' in col_lower:
                                last_name_col = col
                                logger.debug(f"Found last name column for investor replacement: {col}")
                        
                        if first_name_col and last_name_col:
                            # Count empty fields before replacement
                            empty_first_names = filtered_df[first_name_col].isna().sum() + (filtered_df[first_name_col] == '').sum()
                            empty_last_names = filtered_df[last_name_col].isna().sum() + (filtered_df[last_name_col] == '').sum()
                            
                            logger.debug(f"Investor filter: Found {empty_first_names} empty first names and {empty_last_names} empty last names")
                            
                            # Replace empty first names with "Current Owner"
                            filtered_df.loc[filtered_df[first_name_col].isna() | (filtered_df[first_name_col] == ''), first_name_col] = "Current Owner"
                            
                            # Replace empty last names with "Current Owner"
                            filtered_df.loc[filtered_df[last_name_col].isna() | (filtered_df[last_name_col] == ''), last_name_col] = "Current Owner"
                            
                            logger.debug(f"Investor filter: Replaced empty names with 'Current Owner'")
                        
                        logger.debug(f"Investor filter matched {len(filtered_df)} of {len(cleaned_df)} rows")
                    else:
                        filtered_df = cleaned_df  # Default to all data
                else:
                    logger.warning(f"Could not find 'Owner Occupied' column, using all data")
                    logger.debug(f"Available columns: {cleaned_df.columns.tolist()}")
                    filtered_df = cleaned_df
                
                # If filtered DataFrame is empty, return error
                if filtered_df.empty:
                    logger.warning("Filtered data is empty")
                    return JSONResponse(
                        status_code=400,
                        content={"detail": f"No data found for filter type: {filter_type}"}
                    )
                    
                # Use filtered DataFrame for output
                processed_df = filtered_df
            else:
                # No filtering required, use all data
                processed_df = cleaned_df
                
            logger.debug(f"After filtering: {processed_df.shape}")
            
            # Output filename base
            output_filename = f"{file.filename.split('.')[0]}_processed"
            
            # Prepare output based on requested format and mode
            logger.info(f"Preparing output in {output_format} format")
            
            try:
                # Process with multiple sheets for both Excel and CSV
                # Using the same function for consistent output between formats
                sheet_dfs = process_excel_style(processed_df)
                
                if output_format == "excel":
                    # Create Excel file with multiple sheets
                    logger.info(f"Creating Excel file with {len(sheet_dfs)} sheets")
                    
                    # Create a BytesIO buffer to hold the Excel data
                    buffer = BytesIO()
                    
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        # Add each sheet to the Excel file
                        for sheet_name, sheet_df in sheet_dfs.items():
                            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
                            
                            # Get xlsxwriter workbook and worksheet objects
                            workbook = writer.book
                            worksheet = writer.sheets[sheet_name]
                            
                            # Add table formatting
                            table_style = 'Table Style Light 1'
                            if len(sheet_df) > 0:
                                # Create a table with the data
                                worksheet.add_table(0, 0, len(sheet_df), len(sheet_df.columns) - 1, 
                                                 {'name': sheet_name.replace(' ', '_'), 
                                                  'style': table_style})
                    
                    # Get the buffer value
                    buffer.seek(0)
                    processed_data = buffer.getvalue()
                    media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    download_filename = f"{output_filename}.xlsx"
                    
                elif output_format == "csv":
                    # Create a single CSV file with all sheets separated by section headers
                    logger.info(f"Creating single CSV file with {len(sheet_dfs)} sections")
                    
                    # Create a BytesIO buffer to hold the CSV data
                    buffer = BytesIO()
                    
                    # Create a single CSV with sections for each sheet
                    first_sheet = True
                    for sheet_name, sheet_df in sheet_dfs.items():
                        # Add section header (except for first section)
                        if not first_sheet:
                            buffer.write(f"\n\n\n".encode('utf-8'))
                        
                        # Add section header
                        buffer.write(f"### {sheet_name} ###\n".encode('utf-8'))
                        
                        # Add the dataframe as CSV
                        csv_data = sheet_df.to_csv(index=False)
                        buffer.write(csv_data.encode('utf-8'))
                        
                        first_sheet = False
                    
                    # Get the buffer value
                    buffer.seek(0)
                    processed_data = buffer.getvalue()
                    media_type = "text/csv"
                    download_filename = f"{output_filename}.csv"
                    
                else:
                    logger.error(f"Unsupported output format: {output_format}")
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "Unsupported output format"}
                    )
                
                logger.debug(f"Generated file size: {len(processed_data)} bytes")
                
                # Cleanup input file
                if background_tasks:
                    logger.info("Adding cleanup background task")
                    background_tasks.add_task(cleanup_temp_file, [input_path])
                    
                # Return the response with the correct headers
                logger.info(f"Returning processed file: {download_filename}")
                headers = {
                    'Content-Disposition': f'attachment; filename="{download_filename}"',
                    'Content-Type': media_type,
                }
                
                return Response(
                    content=processed_data,
                    headers=headers,
                )
                
            except Exception as e:
                logger.error(f"Error preparing output: {str(e)}")
                logger.error(traceback.format_exc())
                return JSONResponse(
                    status_code=500,
                    content={"detail": f"Error preparing output: {str(e)}"}
                )
            
        except Exception as e:
            logger.error(f"Error cleaning or filtering data: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"detail": f"Error processing data: {str(e)}"}
            )
        
    except Exception as e:
        logger.error(f"Unexpected error processing file: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unexpected error: {str(e)}"}
        )

def cleanup_temp_file(files):
    """Delete temporary files after processing"""
    for file in files:
        try:
            logger.info(f"Cleaning up temporary file: {file}")
            os.unlink(file)
        except Exception as e:
            logger.error(f"Error cleaning up file {file}: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint called")
    return {"status": "ok"}

def process_excel_style(df):
    """
    Process the dataframe according to the Excel macro logic:
    1. Reorganize columns
    2. Format data properly
    3. Create filtered dataframes for different audiences
    
    Returns a dictionary of dataframes for each sheet type
    """
    logger.info("Processing data with Excel-style formatting")
    
    # Make a copy to avoid modifying the original
    master_df = df.copy()
    
    # Step 1: Find or identify the key columns
    mail_owner_col = None
    first_name_col = None
    last_name_col = None
    owner_occupied_col = None
    
    for col in master_df.columns:
        col_lower = col.lower()
        if 'mail owner name' in col_lower:
            mail_owner_col = col
            logger.debug(f"Found Mail Owner column: {col}")
        elif 'owner first name' in col_lower:
            first_name_col = col
            logger.debug(f"Found First Name column: {col}")
        elif 'owner last name' in col_lower:
            last_name_col = col
            logger.debug(f"Found Last Name column: {col}")
        elif 'owner occupied' in col_lower:
            owner_occupied_col = col
            logger.debug(f"Found Owner Occupied column: {col}")
    
    # Create a "Full Name" column from Mail Owner Name with proper case
    if mail_owner_col:
        master_df['Full Name'] = master_df[mail_owner_col].apply(
            lambda x: x.title() if isinstance(x, str) else x
        )
        logger.debug("Created 'Full Name' column with proper case from mail owner name")
    
    # Fill blank cells in first name with full name
    if first_name_col and 'Full Name' in master_df.columns:
        blank_mask = master_df[first_name_col].isna() | (master_df[first_name_col] == '')
        master_df.loc[blank_mask, first_name_col] = master_df.loc[blank_mask, 'Full Name']
        logger.debug(f"Filled {blank_mask.sum()} empty first names with full name")
    
    # Fill blank cells in "Owner Occupied" column with "No"
    if owner_occupied_col:
        blank_mask = master_df[owner_occupied_col].isna() | (master_df[owner_occupied_col] == '')
        master_df.loc[blank_mask, owner_occupied_col] = "No"
        logger.debug(f"Filled {blank_mask.sum()} empty 'Owner Occupied' cells with 'No'")
    
    # Reorder columns if all required columns exist
    if all([mail_owner_col, first_name_col, last_name_col, owner_occupied_col]):
        # Get the list of all columns
        all_cols = list(master_df.columns)
        # Remove the key columns from the list
        for col in [mail_owner_col, first_name_col, last_name_col, owner_occupied_col]:
            if col in all_cols:
                all_cols.remove(col)
        
        # Create new ordered column list
        ordered_cols = [owner_occupied_col, last_name_col, first_name_col, mail_owner_col] + all_cols
        
        # Reorder the DataFrame
        master_df = master_df[ordered_cols]
        logger.debug(f"Reordered columns: {', '.join(ordered_cols[:4])}... are now first")
    
    # Create filtered DataFrames for each sheet
    sheet_dfs = {'Master List': master_df}
    
    # Create Owner Occupied (OO) sheet
    if owner_occupied_col:
        # Find rows where Owner Occupied is "Yes"
        oo_mask = master_df[owner_occupied_col].str.upper().isin(['Y', 'YES', 'TRUE', 'T', '1', 'OWNER', 'OCCUPIED'])
        oo_df = master_df[oo_mask].copy()
        
        # If there are no explicit "Yes" values, try to infer based on non-empty owner names
        if len(oo_df) == 0 and first_name_col:
            oo_mask = master_df[first_name_col].notna() & (master_df[first_name_col] != '')
            oo_df = master_df[oo_mask].copy()
            
        logger.debug(f"Created OO sheet with {len(oo_df)} rows")
        
        if len(oo_df) > 0:
            # Hide Full Name column if it exists (in Excel this would be column A)
            if 'Full Name' in oo_df.columns:
                oo_df = oo_df.drop(columns=['Full Name'])
                
            sheet_dfs['OO'] = oo_df
    
    # Create Investor sheet
    if owner_occupied_col:
        # Find rows where Owner Occupied is "No"
        investor_mask = ~master_df[owner_occupied_col].str.upper().isin(['Y', 'YES', 'TRUE', 'T', '1', 'OWNER', 'OCCUPIED'])
        investor_df = master_df[investor_mask].copy()
        
        logger.debug(f"Created Investor sheet with {len(investor_df)} rows")
        
        if len(investor_df) > 0:
            # Hide Full Name column if it exists
            if 'Full Name' in investor_df.columns:
                investor_df = investor_df.drop(columns=['Full Name'])
            
            # For investor properties, replace empty first and last names with "Current Owner"
            if first_name_col and last_name_col:
                # Count empty fields before replacement
                empty_first_names = investor_df[first_name_col].isna().sum() + (investor_df[first_name_col] == '').sum()
                empty_last_names = investor_df[last_name_col].isna().sum() + (investor_df[last_name_col] == '').sum()
                
                # Replace empty first names with "Current Owner"
                investor_df.loc[investor_df[first_name_col].isna() | (investor_df[first_name_col] == ''), first_name_col] = "Current Owner"
                
                # Replace empty last names with "Current Owner"
                investor_df.loc[investor_df[last_name_col].isna() | (investor_df[last_name_col] == ''), last_name_col] = "Current Owner"
                
                logger.debug(f"Replaced {empty_first_names} empty first names and {empty_last_names} empty last names with 'Current Owner'")
                
            sheet_dfs['Investor'] = investor_df
    
    # Create Renter sheet
    if owner_occupied_col:
        # Find rows where Owner Occupied is "No" (same as Investor)
        renter_mask = ~master_df[owner_occupied_col].str.upper().isin(['Y', 'YES', 'TRUE', 'T', '1', 'OWNER', 'OCCUPIED'])
        renter_df = master_df[renter_mask].copy()
        
        logger.debug(f"Created Renter sheet with {len(renter_df)} rows")
        
        if len(renter_df) > 0:
            # Hide Full Name column if it exists
            if 'Full Name' in renter_df.columns:
                renter_df = renter_df.drop(columns=['Full Name'])
            
            # Replace all First Name cells with "Current Renter"
            if first_name_col:
                renter_df[first_name_col] = "Current Renter"
                logger.debug(f"Set all {len(renter_df)} first names to 'Current Renter'")
            
            # Remove Last Name column if it exists
            if last_name_col in renter_df.columns:
                renter_df = renter_df.drop(columns=[last_name_col])
                logger.debug(f"Removed Last Name column from Renter sheet")
                
            sheet_dfs['Renter'] = renter_df
    
    return sheet_dfs

@app.post("/process_excel_style")
async def process_excel_style_endpoint(
    file: UploadFile = File(...),
    output_format: str = Form(...),
    delimiter: str = Form(','),
    background_tasks: BackgroundTasks = None,
):
    """
    Process a CSV file using Excel-style formatting and return multiple sheets in Excel format or ZIP of CSVs
    """
    logger.info(f"Starting Excel-style processing: {file.filename}, format: {output_format}, delimiter: {delimiter}")
    
    if not file.filename.endswith('.csv'):
        logger.error(f"Invalid file type: {file.filename}")
        return JSONResponse(
            status_code=400,
            content={"detail": "Only CSV files are supported"}
        )
    
    try:
        # Create a unique filename for the processed file
        file_id = str(uuid.uuid4())
        input_path = TEMP_DIR / f"{file_id}_input.csv"
        
        logger.info(f"Saving uploaded file to {input_path}")
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            content = await file.read()
            logger.debug(f"Read {len(content)} bytes from uploaded file")
            buffer.write(content)
        
        # Reset file pointer for future operations
        await file.seek(0)
        
        # Check if file was saved correctly
        if not input_path.exists():
            logger.error(f"Failed to save file to {input_path}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Failed to save uploaded file"}
            )
        
        # Auto-detect delimiter if needed
        if delimiter == 'auto':
            detected_delimiter = detect_delimiter(input_path)
            logger.info(f"Auto-detected delimiter: '{detected_delimiter}'")
            delimiter = detected_delimiter
        
        # Handle special case for tab
        if delimiter == '\\t':
            delimiter = '\t'
        
        # Read CSV with pandas
        logger.info(f"Reading CSV with pandas using delimiter: '{delimiter}'")
        try:
            df = pd.read_csv(input_path, delimiter=delimiter, dtype=str, on_bad_lines='warn')
            logger.debug(f"Read CSV successfully. Shape: {df.shape}, Columns: {df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error reading CSV: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=400,
                content={"detail": f"Error reading CSV: {str(e)}"}
            )
        
        # Clean data
        cleaned_df = clean_data(df)
        
        # Fill empty names
        cleaned_df = fill_empty_names(cleaned_df)
        
        # Process data using Excel-style formatting
        sheet_dfs = process_excel_style(cleaned_df)
        
        # Base output filename
        output_filename = f"{file.filename.split('.')[0]}_processed"
        
        logger.info(f"Creating output with {len(sheet_dfs)} sheets: {', '.join(sheet_dfs.keys())}")
        
        # Process based on requested format
        if output_format == "excel":
            # Create Excel file with multiple sheets
            logger.info("Creating Excel file with multiple sheets")
            
            # Create a BytesIO buffer to hold the Excel data
            buffer = BytesIO()
            
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Add each sheet to the Excel file
                for sheet_name, sheet_df in sheet_dfs.items():
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Get xlsxwriter workbook and worksheet objects
                    workbook = writer.book
                    worksheet = writer.sheets[sheet_name]
                    
                    # Add table formatting
                    table_style = 'Table Style Light 1'
                    if len(sheet_df) > 0:
                        # Create a table with the data
                        worksheet.add_table(0, 0, len(sheet_df), len(sheet_df.columns) - 1, 
                                          {'name': sheet_name.replace(' ', '_'), 
                                          'style': table_style})
            
            # Get the buffer value
            buffer.seek(0)
            processed_data = buffer.getvalue()
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            download_filename = f"{output_filename}.xlsx"
        
        elif output_format == "csv":
            # Create a single CSV file with all sheets separated by section headers
            logger.info("Creating single CSV file with sections for each sheet")
            
            # Create a BytesIO buffer to hold the CSV data
            buffer = BytesIO()
            
            # Create a single CSV with sections for each sheet
            first_sheet = True
            for sheet_name, sheet_df in sheet_dfs.items():
                # Add section header (except for first section)
                if not first_sheet:
                    buffer.write(f"\n\n\n".encode('utf-8'))
                
                # Add section header
                buffer.write(f"### {sheet_name} ###\n".encode('utf-8'))
                
                # Add the dataframe as CSV
                csv_data = sheet_df.to_csv(index=False)
                buffer.write(csv_data.encode('utf-8'))
                
                first_sheet = False
            
            # Get the buffer value
            buffer.seek(0)
            processed_data = buffer.getvalue()
            media_type = "text/csv"
            download_filename = f"{output_filename}.csv"
        
        else:
            logger.error(f"Unsupported output format: {output_format}")
            return JSONResponse(
                status_code=400,
                content={"detail": "Unsupported output format"}
            )
        
        logger.debug(f"Generated file size: {len(processed_data)} bytes")
        
        # Cleanup input file
        if background_tasks:
            background_tasks.add_task(cleanup_temp_file, [input_path])
        
        # Return the response with the correct headers
        headers = {
            'Content-Disposition': f'attachment; filename="{download_filename}"',
            'Content-Type': media_type,
        }
        
        logger.info(f"Returning processed file: {download_filename}")
        return Response(
            content=processed_data,
            headers=headers,
        )
        
    except Exception as e:
        logger.error(f"Unexpected error processing file for Excel style: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unexpected error: {str(e)}"}
        )

@app.get("/templates", response_model=List[TemplateResponse])
async def list_templates():
    """List all available letter templates"""
    logger.info("Listing all letter templates")
    
    templates = []
    for template_id, template in LETTER_TEMPLATES.items():
        templates.append({
            "id": template_id,
            "name": template.get("name", "Unnamed Template"),
            "description": template.get("description", ""),
            "content": template.get("content", ""),
            "created": template.get("created", ""),
            "updated": template.get("updated", "")
        })
    
    logger.debug(f"Returning {len(templates)} templates")
    return templates

@app.get("/templates/{template_id}", response_model=TemplateResponse)
async def get_template(template_id: str):
    """Get a specific letter template by ID"""
    logger.info(f"Getting template with ID: {template_id}")
    
    if template_id not in LETTER_TEMPLATES:
        logger.warning(f"Template not found: {template_id}")
        raise HTTPException(status_code=404, detail="Template not found")
    
    template = LETTER_TEMPLATES[template_id]
    return {
        "id": template_id,
        "name": template.get("name", ""),
        "description": template.get("description", ""),
        "content": template.get("content", ""),
        "created": template.get("created", ""),
        "updated": template.get("updated", "")
    }

@app.post("/templates", response_model=TemplateResponse)
async def create_template(template: TemplateCreate):
    """Create a new letter template"""
    logger.info(f"Creating new template: {template.name}")
    
    try:
        # Generate a unique ID
        template_id = str(uuid.uuid4())
        
        # Convert to dict for storage
        template_dict = template.dict()
        
        # Add timestamps
        import datetime
        now = datetime.datetime.now().isoformat()
        template_dict["created"] = now
        template_dict["updated"] = now
        
        # Save in memory
        LETTER_TEMPLATES[template_id] = template_dict
        
        # Save to disk
        template_path = TEMPLATES_DIR / f"{template_id}.json"
        with open(template_path, "w") as f:
            json.dump(template_dict, f, indent=2)
        
        logger.info(f"Created template with ID: {template_id}")
        return {
            "id": template_id,
            **template_dict
        }
    
    except Exception as e:
        logger.error(f"Error creating template: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error creating template: {str(e)}")

@app.put("/templates/{template_id}", response_model=TemplateResponse)
async def update_template(template_id: str, template: TemplateUpdate):
    """Update an existing letter template"""
    logger.info(f"Updating template with ID: {template_id}")
    
    if template_id not in LETTER_TEMPLATES:
        logger.warning(f"Template not found: {template_id}")
        raise HTTPException(status_code=404, detail="Template not found")
    
    try:
        # Convert to dict for storage
        template_dict = template.dict()
        
        # Update timestamp
        import datetime
        template_dict["updated"] = datetime.datetime.now().isoformat()
        # Preserve creation date
        template_dict["created"] = LETTER_TEMPLATES[template_id].get("created", template_dict["updated"])
        
        # Update in memory
        LETTER_TEMPLATES[template_id] = template_dict
        
        # Save to disk
        template_path = TEMPLATES_DIR / f"{template_id}.json"
        with open(template_path, "w") as f:
            json.dump(template_dict, f, indent=2)
        
        logger.info(f"Updated template: {template_id}")
        return {
            "id": template_id,
            **template_dict
        }
    
    except Exception as e:
        logger.error(f"Error updating template: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error updating template: {str(e)}")

@app.delete("/templates/{template_id}")
async def delete_template(template_id: str):
    """Delete a letter template"""
    logger.info(f"Deleting template with ID: {template_id}")
    
    if template_id not in LETTER_TEMPLATES:
        logger.warning(f"Template not found: {template_id}")
        raise HTTPException(status_code=404, detail="Template not found")
    
    try:
        # Remove from memory
        del LETTER_TEMPLATES[template_id]
        
        # Remove from disk
        template_path = TEMPLATES_DIR / f"{template_id}.json"
        if template_path.exists():
            os.unlink(template_path)
        
        logger.info(f"Deleted template: {template_id}")
        return {"success": True, "id": template_id}
    
    except Exception as e:
        logger.error(f"Error deleting template: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error deleting template: {str(e)}")

@app.get("/template_variables")
async def get_template_variables():
    """Get available template variables"""
    logger.info("Getting template variables")
    return TEMPLATE_VARIABLES

@app.post("/generate_letters")
async def generate_letters(
    file: UploadFile = File(...),
    template_id: str = Form(...),
    filter_type: str = Form("all"),
    delimiter: str = Form(','),
    background_tasks: BackgroundTasks = None,
):
    """
    Generate letters from a CSV file and a letter template
    """
    logger.info(f"Generating letters from template {template_id} with filter {filter_type}")
    
    if template_id not in LETTER_TEMPLATES:
        logger.warning(f"Template not found: {template_id}")
        return JSONResponse(
            status_code=404,
            content={"detail": "Template not found"}
        )
    
    template = LETTER_TEMPLATES[template_id]
    
    try:
        # Create a unique filename for the processed file
        file_id = str(uuid.uuid4())
        input_path = TEMP_DIR / f"{file_id}_input.csv"
        
        logger.info(f"Saving uploaded file to {input_path}")
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            content = await file.read()
            logger.debug(f"Read {len(content)} bytes from uploaded file")
            buffer.write(content)
        
        # Reset file pointer for future operations
        await file.seek(0)
        
        # Auto-detect delimiter if needed
        if delimiter == 'auto':
            detected_delimiter = detect_delimiter(input_path)
            logger.info(f"Auto-detected delimiter: '{detected_delimiter}'")
            delimiter = detected_delimiter
        
        # Handle special case for tab
        if delimiter == '\\t':
            delimiter = '\t'
        
        # Read CSV with pandas
        logger.info(f"Reading CSV with pandas using delimiter: '{delimiter}'")
        try:
            df = pd.read_csv(input_path, delimiter=delimiter, dtype=str, on_bad_lines='warn')
            logger.debug(f"Read CSV successfully. Shape: {df.shape}, Columns: {df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error reading CSV: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=400,
                content={"detail": f"Error reading CSV: {str(e)}"}
            )
        
        # Clean data and fill empty names
        cleaned_df = clean_data(df)
        cleaned_df = fill_empty_names(cleaned_df)
        
        # Apply filtering
        if filter_type != "all":
            # Apply the same filter logic from the upload endpoint
            # This ensures consistency between filters used for CSV processing and letter generation
            # Check if "Owner Occupied" column exists
            owner_occupied_col = None
            for col in cleaned_df.columns:
                if "owner occupied" in col.lower():
                    owner_occupied_col = col
                    logger.debug(f"Found Owner Occupied column: {col}")
                    break
            
            # Find relevant name/address columns
            first_name_col = None
            last_name_col = None
            address_col = None
            city_col = None
            state_col = None
            zip_col = None
            
            for col in cleaned_df.columns:
                col_lower = col.lower()
                if 'first name' in col_lower:
                    first_name_col = col
                elif 'last name' in col_lower:
                    last_name_col = col
                elif 'address' in col_lower and not any(x in col_lower for x in ['city', 'state', 'zip']):
                    address_col = col
                elif 'city' in col_lower:
                    city_col = col
                elif 'state' in col_lower:
                    state_col = col
                elif 'zip' in col_lower or 'postal' in col_lower:
                    zip_col = col
            
            if owner_occupied_col:
                if filter_type == "owner_occupied":
                    # Owner occupied = yes filtering logic
                    has_positive_values = any(val.upper().strip() in ["Y", "YES", "TRUE", "T", "1", "OWNER", "OCCUPIED"] 
                                          for val in cleaned_df[owner_occupied_col].fillna('') if isinstance(val, str))
                    
                    owner_occupied_mask = cleaned_df[owner_occupied_col].str.upper().fillna('').str.strip().isin(["Y", "YES", "TRUE", "T", "1", "OWNER", "OCCUPIED"])
                    # If no records match our positive value check, assume field has different format
                    if not owner_occupied_mask.any() and not has_positive_values:
                        logger.warning("No records match positive values in Owner Occupied column")
                        # Try a different approach - look for non-empty values in first name column
                        if first_name_col:
                            owner_occupied_mask = cleaned_df[first_name_col].notna() & (cleaned_df[first_name_col] != '')
                    
                    filtered_df = cleaned_df[owner_occupied_mask]
                    
                elif filter_type == "renter":
                    # Renter filtering logic
                    has_negative_values = any(val.upper().strip() in ["N", "NO", "FALSE", "F", "0", "RENTER"] 
                                         for val in cleaned_df[owner_occupied_col].fillna('') if isinstance(val, str))
                    
                    renter_mask = cleaned_df[owner_occupied_col].str.upper().fillna('').str.strip().isin(["N", "NO", "FALSE", "F", "0", "RENTER"])
                    
                    # If no records match our negative value check, assume empty means not owner occupied
                    if not renter_mask.any() and not has_negative_values:
                        renter_mask = cleaned_df[owner_occupied_col].isna() | (cleaned_df[owner_occupied_col] == '')
                    
                    filtered_df = cleaned_df[renter_mask].copy()
                    
                    # Replace owner name fields with "Current Renter"
                    if first_name_col:
                        filtered_df[first_name_col] = "Current Renter"
                    if last_name_col:
                        filtered_df[last_name_col] = ""
                        
                elif filter_type == "investor":
                    # Investor filtering logic
                    investor_mask = cleaned_df[owner_occupied_col].str.upper().fillna('').str.strip().isin(["N", "NO", "FALSE", "F", "0", "RENTER"])
                    
                    # If no explicit negative values, consider empty values as not owner occupied
                    if not investor_mask.any():
                        investor_mask = cleaned_df[owner_occupied_col].isna() | (cleaned_df[owner_occupied_col] == '')
                        
                    # Additional check: empty values only count as investor if we have owner name data
                    if first_name_col:
                        has_owner_data = filtered_df[first_name_col].notna().any()
                        if has_owner_data:
                            non_empty_owner = cleaned_df[first_name_col].notna() & (cleaned_df[first_name_col] != '')
                            investor_mask = investor_mask & non_empty_owner
                    
                    filtered_df = cleaned_df[investor_mask].copy()
                    
                    # For investor properties, replace empty first and last names with "Current Owner"
                    if first_name_col:
                        filtered_df.loc[filtered_df[first_name_col].isna() | (filtered_df[first_name_col] == ''), first_name_col] = "Current Owner"
                    if last_name_col:
                        filtered_df.loc[filtered_df[last_name_col].isna() | (filtered_df[last_name_col] == ''), last_name_col] = "Current Owner"
                
                else:
                    filtered_df = cleaned_df  # Default to all data
            else:
                logger.warning(f"Could not find 'Owner Occupied' column, using all data")
                filtered_df = cleaned_df
                
            # If filtered DataFrame is empty, return error
            if filtered_df.empty:
                logger.warning("Filtered data is empty")
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"No data found for filter type: {filter_type}"}
                )
                
            # Use filtered DataFrame for output
            processed_df = filtered_df
        else:
            # No filtering required, use all data
            processed_df = cleaned_df
        
        # Get column indices for required fields
        column_names = processed_df.columns.tolist()
        
        # Identify required columns
        first_name_idx = next((i for i, col in enumerate(column_names) if 'first name' in col.lower()), None)
        last_name_idx = next((i for i, col in enumerate(column_names) if 'last name' in col.lower()), None)
        address_idx = next((i for i, col in enumerate(column_names) if 'address' in col.lower() and not any(x in col.lower() for x in ['city', 'state', 'zip'])), None)
        city_idx = next((i for i, col in enumerate(column_names) if 'city' in col.lower()), None)
        state_idx = next((i for i, col in enumerate(column_names) if 'state' in col.lower()), None)
        zip_idx = next((i for i, col in enumerate(column_names) if 'zip' in col.lower() or 'postal' in col.lower()), None)
        
        # Generate letters for each record
        letters = []
        import datetime
        current_date = datetime.datetime.now().strftime("%m/%d/%Y")
        current_month = datetime.datetime.now().strftime("%B")
        current_year = datetime.datetime.now().strftime("%Y")
        
        # Get template content
        template_content = template.get("content", "")
        
        # Process each row
        for _, row in processed_df.iterrows():
            # Create a variable mapping for this record
            variables = {}
            
            # Extract fields from the dataframe
            if first_name_idx is not None:
                variables["{first_name}"] = row.iloc[first_name_idx]
            
            if last_name_idx is not None:
                variables["{last_name}"] = row.iloc[last_name_idx]
                
                # Create full name if we have both parts
                if first_name_idx is not None:
                    first_name = row.iloc[first_name_idx]
                    last_name = row.iloc[last_name_idx]
                    if first_name and last_name:
                        if first_name.lower() == "current renter":
                            variables["{full_name}"] = "Current Renter"
                        elif first_name.lower() == "current owner":
                            variables["{full_name}"] = "Current Owner"
                        else:
                            variables["{full_name}"] = f"{first_name} {last_name}"
                    elif first_name and not last_name:
                        variables["{full_name}"] = first_name
                    elif not first_name and last_name:
                        variables["{full_name}"] = last_name
            
            if address_idx is not None:
                variables["{address}"] = row.iloc[address_idx]
            
            if city_idx is not None:
                variables["{city}"] = row.iloc[city_idx]
            
            if state_idx is not None:
                variables["{state}"] = row.iloc[state_idx]
            
            if zip_idx is not None:
                variables["{zip}"] = row.iloc[zip_idx]
            
            # Add special variables
            if first_name_idx is not None and row.iloc[first_name_idx] == "Current Renter":
                variables["{current_renter}"] = "Current Renter"
            else:
                variables["{current_renter}"] = ""
                
            if first_name_idx is not None and row.iloc[first_name_idx] == "Current Owner":
                variables["{current_owner}"] = "Current Owner"
            else:
                variables["{current_owner}"] = ""
            
            # Add date variables
            variables["{current_date}"] = current_date
            variables["{current_month}"] = current_month
            variables["{current_year}"] = current_year
            
            # Generate personalized letter content by replacing variables
            letter_content = template_content
            for var, value in variables.items():
                if var in letter_content:
                    if value:  # Only replace if we have a value
                        letter_content = letter_content.replace(var, value)
            
            # Append to the list of letters
            letters.append({
                "content": letter_content,
                "variables": variables
            })
        
        # Create a response with the generated letters
        return {
            "template_name": template.get("name", "Unnamed Template"),
            "filter_type": filter_type,
            "record_count": len(letters),
            "letters": letters
        }
        
    except Exception as e:
        logger.error(f"Error generating letters: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error generating letters: {str(e)}"}
        )

@app.post("/print_letters")
async def print_letters(letters_request: LettersRequest):
    """
    Generate a printable document from letter contents
    """
    logger.info(f"Creating printable output with {len(letters_request.letters)} letters in {letters_request.output_format} format")
    
    try:
        # Create a unique filename for the output
        output_id = str(uuid.uuid4())
        output_filename = f"letters_{output_id}"
        
        # Currently only supporting HTML format for browser printing
        if letters_request.output_format == "html":
            # Create HTML content with all letters
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Real Estate Letters</title>
                <style>
                    @media print {{
                        .letter {{
                            page-break-after: always;
                            margin: 0;
                            padding: 0.5in;
                        }}
                    }}
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.5;
                        margin: 0;
                    }}
                    .letter {{
                        width: 8.5in;
                        min-height: 11in;
                        padding: 0.5in;
                        position: relative;
                        box-sizing: border-box;
                    }}
                    .letter-content {{
                        white-space: pre-wrap;
                    }}
                    .letter-footer {{
                        position: absolute;
                        bottom: 0.5in;
                        width: calc(100% - 1in);
                        text-align: center;
                        font-size: 0.8em;
                        color: #666;
                    }}
                    @media screen {{
                        .letter {{
                            margin: 20px auto;
                            border: 1px solid #ccc;
                            box-shadow: 0 0 10px rgba(0,0,0,0.1);
                        }}
                    }}
                </style>
            </head>
            <body>
            """
            
            # Add each letter
            for i, letter in enumerate(letters_request.letters):
                html_content += f"""
                <div class="letter">
                    <div class="letter-content">{letter.content}</div>
                    <div class="letter-footer">Letter {i+1} of {len(letters_request.letters)}</div>
                </div>
                """
            
            html_content += """
            <script>
                // Auto-print when page loads
                window.onload = function() {
                    // setTimeout to give the browser time to render the content
                    setTimeout(function() {
                        window.print();
                    }, 500);
                };
            </script>
            </body>
            </html>
            """
            
            # Save the HTML file
            output_path = TEMP_DIR / f"{output_filename}.html"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            # Return the file path
            return FileResponse(
                path=output_path,
                filename=f"{output_filename}.html",
                media_type="text/html"
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported output format: {letters_request.output_format}")
        
    except Exception as e:
        logger.error(f"Error creating printable output: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error creating printable output: {str(e)}")

@app.get("/test")
async def test_endpoint():
    """Test API endpoint"""
    logger.info("Test endpoint called")
    return {"status": "ok", "message": "API is working"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 