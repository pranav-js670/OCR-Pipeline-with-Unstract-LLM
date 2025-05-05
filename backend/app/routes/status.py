from fastapi import APIRouter, HTTPException
from app.models import StatusResponse
from app.services.unstract import check_status

router = APIRouter()


@router.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str):
    try:
        status = check_status(job_id)
        return {"status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
