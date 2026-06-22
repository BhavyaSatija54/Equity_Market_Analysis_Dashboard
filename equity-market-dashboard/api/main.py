"""
api/main.py
------------
FastAPI application entry point.
Configures CORS, middleware, lifespan (startup/shutdown),
and mounts all routers.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.routes import analytics, equities


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("Starting Equity Market Analysis API")
    # In production: initialise Spark session, load data into memory cache
    yield
    logger.info("Shutting down API")


def create_app() -> FastAPI:
    try:
        with open("config/config.yaml") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}

    app_cfg = config.get("app", {})

    app = FastAPI(
        title="Equity Market Analysis Dashboard",
        description=(
            "REST API for the Equity Market Analysis platform. "
            "Provides equity screener, risk metrics, scenario analysis, "
            "and automated reporting endpoints."
        ),
        version=app_cfg.get("version", "1.0.0"),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    cors_origins = config.get("api", {}).get("cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}s"
        return response

    # Global error handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # Routers
    app.include_router(equities.router,  prefix="/api/v1/equities",  tags=["Equities"])
    app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])

    # Health check
    @app.get("/health", tags=["System"])
    async def health():
        return {"status": "ok", "version": app_cfg.get("version", "1.0.0")}

    return app


app = create_app()
