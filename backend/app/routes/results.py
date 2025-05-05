# from fastapi import APIRouter, HTTPException
# from app.models import ResultResponse
# from app.services.unstract import retrieve_text
# from app.services.llm import parse_health_report

# router = APIRouter()


# @router.get("/results/{job_id}", response_model=ResultResponse)
# def get_results(job_id: str):
#     try:
#         raw_text = retrieve_text(job_id)
#         parameters = parse_health_report(raw_text)
#         return {"parameters": parameters}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException
from starlette.responses import JSONResponse
from app.models import ResultResponse
from app.services.unstract import retrieve_text
from app.services.llm import parse_health_report
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/results/{job_id}", response_model=ResultResponse)
def get_results(job_id: str):
    try:
        # 1) pull raw text
        raw_text = retrieve_text(job_id)
        # 2) call the LLM to parse into list-of-dicts
        parameters = parse_health_report(raw_text)

        # Safety: ensure it's a list of dicts
        if not isinstance(parameters, list):
            raise ValueError(f"Expected list, got {type(parameters).__name__}")
        for idx, p in enumerate(parameters):
            if not isinstance(p, dict):
                raise ValueError(f"Parameter at index {idx} is not a dict: {p!r}")

        return {"parameters": parameters}

    except HTTPException:
        # If you already raised an HTTPException upstream, re-raise it
        raise
    except Exception as e:
        # Log full stack for your debugging
        logger.exception(f"Error in /results/{job_id}")
        # Return JSON with error detail instead of a bare 500
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to produce results", "detail": str(e)},
        )
