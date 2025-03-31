# Real Estate Data Processor

This application helps process real estate CSV data by standardizing formatting and allowing export to different file formats.

## Features

- Upload CSV files with real estate data
- Process and standardize data formatting:
  - Convert column values to consistent formats
  - Standardize ZIP codes to 5 digits with leading zeros
  - Format phone numbers consistently
  - Clean price fields (remove $ and commas)
- Export to multiple formats:
  - CSV
  - Excel
  - Numbers (via Excel format)

## Project Structure

```
re-letters/
├── backend/               # Python FastAPI backend
│   ├── main.py            # Main API code
│   ├── requirements.txt   # Python dependencies
│   └── Dockerfile         # Docker configuration
├── src/                   # Next.js frontend
│   ├── app/               # Next.js app router
│   ├── components/        # React components
│   └── lib/               # Utility functions and types
```

## Getting Started

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd re-letters/backend
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the backend server:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Using Docker

1. Build and run using Docker Compose:
   ```
   cd re-letters/backend
   docker-compose up
   ```

### Frontend Setup

1. Navigate to the project root:
   ```
   cd re-letters
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Run the development server:
   ```
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Usage

1. Upload a CSV file using the file uploader
2. Select your desired output format (CSV, Excel, or Numbers)
3. Click "Process File"
4. Once processing is complete, download the processed file

## Technologies Used

- **Backend**:
  - FastAPI
  - Pandas for data processing
  - Docker for containerization

- **Frontend**:
  - Next.js
  - React
  - Tailwind CSS
  - shadcn/ui components 