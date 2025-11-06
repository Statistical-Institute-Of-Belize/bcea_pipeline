import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import uvicorn

# Import routers
from .routers import predict
from .config import settings
from .dependencies import get_model_and_tokenizer

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(predict.router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirects to the API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if model can be loaded
        model, tokenizer, label_map, bcea_descriptions = get_model_and_tokenizer()
        
        # Check if model is loaded successfully
        if model is None or tokenizer is None:
            return {"status": "error", "message": "Model or tokenizer failed to load"}
        
        # Return success
        return {
            "status": "ok",
            "model_loaded": True,
            "classes": len(label_map) if label_map else 0,
            "bcea_descriptions": len(bcea_descriptions) if bcea_descriptions else 0
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def start():
    """Run the API server using uvicorn."""
    # Start uvicorn server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    start()
