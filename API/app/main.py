import logging
import os
import sys
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.db.database import create_tables
from app.routers.prediction import router

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("insurance_api")

app = FastAPI(
    title="Insurance Prediction API",
    description="Moteur de tarification assurance auto — fréquence, sévérité, prime pure",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://insurance-frontend-ruby.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    logger.info("Démarrage de l'API — création des tables si nécessaire")
    create_tables()
    logger.info("API prête")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"← {request.method} {request.url.path} {response.status_code}")
    return response


app.include_router(router)

DEBUG = os.getenv("DEBUG", "true").lower() == "true"


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Erreur non gérée sur {request.url.path}: {exc}", exc_info=True)
    if DEBUG:
        return JSONResponse(
            status_code=500,
            content={
                "error": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error"},
    )