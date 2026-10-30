"""FastAPI application for the LFM3 Neuron Explorer."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.core.settings import settings
from backend.app.services.model_loader import ModelManager


def create_app(manager: ModelManager | None = None) -> FastAPI:
    app = FastAPI(
        title="LFM3 Neuron Explorer",
        version="0.1.0",
        description="Inspect and steer activations of LiquidAI/LFM3-1.7B during inference.",
    )
    app.state.manager = manager or ModelManager()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    if settings.frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
