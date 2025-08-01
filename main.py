from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from routers import query_router, hackrx_router
from config.settings import get_settings

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting HackRx API...")
    # Initialize services here if needed
    yield
    # Shutdown
    print("🛑 Shutting down HackRx API...")

# FastAPI app initialization
app = FastAPI(
    title="HackRx API",
    description="AI-powered multimodal document processing with RAG pipeline for insurance decisions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query_router.router, prefix="/api", tags=["query"])
app.include_router(hackrx_router.router, prefix="/hackrx", tags=["hackrx"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRx API",
        "version": "1.0.0",
        "endpoints": {
            "hackrx_run": "/hackrx/run",
            "hackrx_health": "/hackrx/health",
            "query": "/api/query",
            "health": "/api/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import os
    
    # Check if SSL certificates exist
    ssl_keyfile = "key.pem" if os.path.exists("key.pem") else None
    ssl_certfile = "cert.pem" if os.path.exists("cert.pem") else None
    
    # Run with optional HTTPS support
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        ssl_keyfile_password=None
    )
