"""Version 1 API routes."""

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.schemas import AnalyzeRequest, AnalyzeResponse
from app.data.repositories import UnknownAccountError, UnknownBankError
from app.services.orchestrator import IntelligenceOrchestrator

router = APIRouter(prefix="/api/v1")


def get_orchestrator(request: Request) -> IntelligenceOrchestrator:
    existing = getattr(request.app.state, "orchestrator", None)
    if existing is not None:
        return existing
    # Startup is lazy so the health endpoint remains useful if an artifact is missing.
    with request.app.state.orchestrator_lock:
        existing = getattr(request.app.state, "orchestrator", None)
        if existing is None:
            try:
                existing = IntelligenceOrchestrator()
            except (FileNotFoundError, ValueError, TypeError) as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            request.app.state.orchestrator = existing
    return existing


@router.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze(
    body: AnalyzeRequest,
    orchestrator: IntelligenceOrchestrator = Depends(get_orchestrator),
) -> AnalyzeResponse:
    try:
        return orchestrator.analyze(body)
    except UnknownBankError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UnknownAccountError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
