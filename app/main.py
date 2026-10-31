"""Main FastAPI Application Entrypoint.

Configures application routes, static assets, templates, CORS, and lifecycle events.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.config import settings

# Setup standard logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("document_ai_assistant")

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown procedures."""
    logger.info(f"Starting {settings.APP_NAME}...")
    logger.info(f"Embeddings Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"Ollama Endpoint: {settings.OLLAMA_BASE_URL} (Model: {settings.OLLAMA_MODEL})")
    logger.info(f"ChromaDB persist directory: {settings.CHROMA_PERSIST_DIR}")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Resume-grade Document AI Assistant powered by local Retrieval-Augmented Generation (RAG).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and Template mounts
static_path = BASE_DIR / "static"
templates_path = BASE_DIR / "templates"

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
templates = Jinja2Templates(directory=str(templates_path))

from app.dependencies import get_generation_service

# Include API Routers
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/api/models", tags=["LLM"])
async def list_models():
    """List available local Ollama models and currently active default."""
    gen_service = get_generation_service(settings)
    info = await gen_service.check_availability()
    return {
        "models": info.get("available_models", []),
        "default_model": settings.OLLAMA_MODEL,
        "online": info.get("online", False),
    }


@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def serve_index(request: Request):
    """Serve the single-page application frontend with pre-loaded models."""
    gen_service = get_generation_service(settings)
    ollama_info = await gen_service.check_availability()
    models = ollama_info.get("available_models", [])

    if settings.OLLAMA_MODEL and settings.OLLAMA_MODEL not in models:
        models = [settings.OLLAMA_MODEL] + models

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": settings.APP_NAME,
            "ollama_model": settings.OLLAMA_MODEL,
            "available_models": models,
            "embedding_model": settings.EMBEDDING_MODEL,
        },
    )
