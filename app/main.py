from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import cv

app = FastAPI(
    title=settings.APP_NAME,
    description="CV Reader API - Classify and extract text from CVs",
    version="1.0.0"
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
app.include_router(cv.router, prefix="/api/cv", tags=["CV"])


@app.get("/")
def root():
    return {"message": "CV Reader API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
