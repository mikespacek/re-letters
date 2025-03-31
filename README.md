# Real Estate Letters Generator

A web application for creating customized real estate letters using CSV data and letter templates.

## Features

- Upload and process CSV files with real estate data
- Create and manage letter templates with variable placeholders
- Generate customized letters by merging CSV data with templates
- Filter properties by owner type (owner-occupied, renter, investor)
- Download generated letters for printing

## Project Structure

- `main.py` - FastAPI backend API
- `test.html` - CSV upload and processing interface
- `letters.html` - Letter templates and generation interface
- `templates/` - Directory for saving letter templates
- `temp/` - Temporary directory for file storage

## Deployment on Vercel

This application is ready to deploy on Vercel. Here's how to deploy it:

1. **Fork or Clone this Repository**
   - Make sure you have a GitHub account
   - Fork this repository or push it to your own GitHub repository

2. **Connect to Vercel**
   - Go to [Vercel](https://vercel.com) and sign up/login
   - Click "Add New" → "Project"
   - Import your GitHub repository
   - Vercel will automatically detect the configuration

3. **Configure Environment Variables (if needed)**
   - Set any required environment variables in the Vercel dashboard

4. **Deploy**
   - Click "Deploy"
   - Vercel will build and deploy your application
   - You'll receive a URL for your deployed application

## Local Development

To run the application locally:

1. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the FastAPI backend:
   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8001
   ```

3. Serve the HTML files:
   ```bash
   python -m http.server 3005
   ```

4. Access the application:
   - CSV Processor: http://localhost:3005/test.html
   - Letters Generator: http://localhost:3005/letters.html
   - API: http://localhost:8001

## API Endpoints

- `/health` - Health check endpoint
- `/templates` - List, create, update, delete letter templates
- `/upload` - Upload and process CSV files
- `/download` - Download processed CSV data
- `/process_excel_style` - Process CSV data in Excel format
- `/print_letters` - Generate HTML letters

## License

MIT 