from fastapi import APIRouter, File, UploadFile, HTTPException
from app.models import UploadResponse
from app.services.unstract import submit_document

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    try:
        job_id = submit_document(file_bytes=content, filename=file.filename)
        return {"job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
