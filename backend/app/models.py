from pydantic import BaseModel
from typing import List


class UploadResponse(BaseModel):
    job_id: str


class StatusResponse(BaseModel):
    status: str


class Parameter(BaseModel):
    parameter: str
    value: float
    unit: str
    reference_range: List[float]
    assessment: str


class ResultResponse(BaseModel):
    parameters: List[Parameter]


# backend/app/models.py

from pydantic import BaseModel, Field
from typing import List, Optional


class LabParameter(BaseModel):
    parameter: str = Field(..., description="Name of the lab test")
    value: float = Field(..., description="Numeric result")
    unit: Optional[str] = Field(None, description="Unit of measurement, if present")
    reference_range: List[Optional[float]] = Field(
        ..., description="Two-element list [low, high], with null for missing ends"
    )
    assessment: Optional[str] = Field(
        None, description="One of 'low','normal','high', if provided"
    )


class ResultResponse(BaseModel):
    parameters: List[LabParameter]
