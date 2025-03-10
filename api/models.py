from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union


class BusinessInput(BaseModel):
    """Single business input for BCEA classification."""
    bus_name: str = Field(..., description="Business name to classify", min_length=1)
    description: str = Field(..., description="Business description", min_length=1)


class BusinessBatchInput(BaseModel):
    """Batch of businesses for BCEA classification."""
    businesses: List[BusinessInput] = Field(..., description="List of businesses to classify", min_items=1)


class AlternativePrediction(BaseModel):
    """Alternative prediction with code, description and confidence."""
    code: str = Field(..., description="BCEA code")
    description: Optional[str] = Field(None, description="Industry description from reference data")
    confidence: float = Field(..., description="Confidence score (0-1)")


class BCEAPrediction(BaseModel):
    """BCEA prediction output."""
    bus_name: str = Field(..., description="Original business name")
    description: str = Field(..., description="Original business description")
    predicted_code: str = Field(..., description="Predicted BCEA code")
    industry_description: Optional[str] = Field(None, description="Industry description from reference data")
    confidence: float = Field(..., description="Confidence score (0-1)")
    confidence_grade: str = Field(..., description="Confidence grade (very_low, low, medium, high, very_high)")
    alternatives: List[AlternativePrediction] = Field(
        default_factory=list, 
        description="Alternative predictions (typically the next best 2 predictions)"
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    predictions: List[BCEAPrediction]