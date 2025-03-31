# Deployment Instructions for Vercel

This project has been configured for deployment on Vercel. Follow these steps to deploy:

## 1. GitHub Repository Setup

First, push this code to GitHub:

```bash
# If you haven't already initialized git
git init

# Add all files
git add .

# Commit the changes
git commit -m "Prepare for Vercel deployment: remove pandas dependency"

# Add GitHub remote (replace with your repository URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

## 2. Deploy on Vercel

1. Go to [Vercel](https://vercel.com) and sign up/login
2. Click "Add New" → "Project"
3. Import your GitHub repository
4. In the configuration step:
   - Keep all the default settings
   - Vercel will detect the project as a Python application
   - The build settings are already defined in the vercel.json file
5. Click "Deploy"

## 3. Using Your Deployed Application

Once deployed, you can access your application at the URL provided by Vercel:

- Frontend: `https://your-project-name.vercel.app/test.html` or `https://your-project-name.vercel.app/letters.html`
- API: `https://your-project-name.vercel.app/health` (to check if the API is working)
- Debug: `https://your-project-name.vercel.app/vercel_test.html` (to test API connectivity)
- Minimal API: `https://your-project-name.vercel.app/api/minimal` (for debugging)
- Debug API: `https://your-project-name.vercel.app/api/debug` (for environment details)

## Troubleshooting

If you encounter any issues:

1. Check the Vercel deployment logs for errors
2. Verify that the API endpoints are working by accessing the debug endpoints first:
   - `/api/minimal` - A minimal API with no imports
   - `/api/debug` - An API that shows environment details
   - `/health` - The regular health check endpoint

3. Common issues and solutions:
   - **404 NOT_FOUND errors**: Check that all routes are correctly defined in vercel.json
   - **500 Internal Server errors**: Check the deployment logs for import or dependency issues
   - **API works but frontend can't access it**: Check for CORS issues or network errors

4. Make sure all files were correctly pushed to GitHub including:
   - api/index.py
   - api/minimal.py
   - api/vercel_debug.py
   - main.py
   - requirements.txt
   - vercel.json
   - vercel_test.html

## Advanced Troubleshooting

If you're still having issues, you can:

1. Try accessing the `/api/debug` endpoint to check the environment
2. Check if static files are being served by accessing `vercel_test.html`
3. Test the minimal API endpoint first before the more complex ones
4. Consider setting up environment variables in the Vercel dashboard if needed 