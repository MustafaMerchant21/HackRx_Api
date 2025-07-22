from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from routers import query_router
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
    description="AI-powered document processing with RAG pipeline for insurance decisions",
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

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRx API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/api/query",
            "health": "/api/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
