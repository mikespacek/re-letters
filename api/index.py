try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import sys
    import os
    
    # Try to import from main, but provide fallback if it fails
    try:
        from main import app
        print("Successfully imported app from main.py")
    except Exception as e:
        print(f"Error importing from main.py: {str(e)}")
        # Create a fallback app if main import fails
        app = FastAPI()
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint that works even if main.py fails"""
            return {"status": "ok", "message": "Fallback health check"}
        
        @app.get("/test")
        async def test_endpoint():
            """Test endpoint that works even if main.py fails"""
            return {"status": "ok", "message": "Fallback test endpoint"}
    
    # Add error handlers
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "type": str(type(exc).__name__)},
        )
    
    # Make handler available for Vercel serverless
    handler = app

except Exception as e:
    # Last resort handler if everything fails
    from fastapi import FastAPI
    app = FastAPI()
    
    @app.get("/health")
    async def emergency_health():
        return {"status": "error", "message": f"Emergency fallback activated. Error: {str(e)}"}
    
    handler = app 