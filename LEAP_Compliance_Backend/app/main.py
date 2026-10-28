"""FastAPI application entry point."""

from threading import Lock

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="AI-Powered Cross-Domain Compliance Intelligence",
    version="0.1.0",
)
app.state.orchestrator = None
app.state.orchestrator_lock = Lock()
app.include_router(router)
