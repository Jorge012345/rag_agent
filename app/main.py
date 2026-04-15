"""FastAPI application entry point."""

from fastapi import FastAPI

from app.interfaces.api.v1.router import router as api_v1_router

app = FastAPI(
    title="Project2 API",
    description="API for document upload and processing",
    version="1.0.0",
)

app.include_router(api_v1_router)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Project2 API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
