from fastapi import FastAPI
from app.routes import upload, status, results

app = FastAPI(title="Health OCR and Analysis API")

app.include_router(upload.router, prefix="/api")
app.include_router(status.router, prefix="/api")
app.include_router(results.router, prefix="/api")
