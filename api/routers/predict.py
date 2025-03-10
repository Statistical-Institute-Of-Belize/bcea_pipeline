from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional, Dict
import pandas as pd
import tempfile
import io
import os
from datetime import datetime

from ..dependencies import get_model_and_tokenizer
from ..models import (
    BusinessInput, 
    BusinessBatchInput,
    BCEAPrediction, 
    BatchPredictionResponse
)
from ..prediction_service import predict_single_business, predict_batch_businesses

router = APIRouter(
    prefix="/predict",
    tags=["prediction"],
    responses={404: {"description": "Not found"}},
)


@router.post("/business", response_model=BCEAPrediction)
async def predict_business(
    business: BusinessInput,
    model_data=Depends(get_model_and_tokenizer)
):
    """
    Predict BCEA code for a single business.
    """
    model, tokenizer, label_map, bcea_descriptions = model_data
    
    prediction = predict_single_business(
        business=business,
        model=model,
        tokenizer=tokenizer,
        label_map=label_map,
        bcea_descriptions=bcea_descriptions
    )
    
    return prediction


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch: BusinessBatchInput,
    model_data=Depends(get_model_and_tokenizer)
):
    """
    Predict BCEA codes for a batch of businesses.
    """
    model, tokenizer, label_map, bcea_descriptions = model_data
    
    predictions = predict_batch_businesses(
        businesses=batch.businesses,
        model=model,
        tokenizer=tokenizer,
        label_map=label_map,
        bcea_descriptions=bcea_descriptions
    )
    
    return BatchPredictionResponse(predictions=predictions)


@router.post("/csv", response_class=FileResponse)
async def predict_from_csv(
    file: UploadFile = File(...),
    model_data=Depends(get_model_and_tokenizer)
):
    """
    Predict BCEA codes from a CSV file.
    The CSV file must have columns 'bus_name' and 'description'.
    Returns a CSV file with predictions added.
    """
    model, tokenizer, label_map, bcea_descriptions = model_data
    
    # Check file format
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a CSV file."
        )
    
    # Read CSV file
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {str(e)}"
        )
    
    # Check required columns
    required_columns = ['bus_name', 'description']
    if not all(col in df.columns for col in required_columns):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV file must contain columns: {', '.join(required_columns)}"
        )
    
    # Convert dataframe to list of BusinessInput objects
    businesses = [
        BusinessInput(bus_name=row['bus_name'], description=row['description'])
        for _, row in df.iterrows()
    ]
    
    # Get predictions
    predictions = predict_batch_businesses(
        businesses=businesses,
        model=model,
        tokenizer=tokenizer,
        label_map=label_map,
        bcea_descriptions=bcea_descriptions
    )
    
    # Add predictions to dataframe
    for i, pred in enumerate(predictions):
        df.loc[i, 'bcea_code'] = pred.predicted_code
        df.loc[i, 'industry_description'] = pred.industry_description
        df.loc[i, 'confidence'] = pred.confidence
        df.loc[i, 'confidence_grade'] = pred.confidence_grade
        
        # Add alternatives
        if len(pred.alternatives) > 0:
            df.loc[i, 'alt_bcea_code_1'] = pred.alternatives[0].code
            df.loc[i, 'alt_industry_description_1'] = pred.alternatives[0].description
            df.loc[i, 'alt_confidence_1'] = pred.alternatives[0].confidence
            
        if len(pred.alternatives) > 1:
            df.loc[i, 'alt_bcea_code_2'] = pred.alternatives[1].code
            df.loc[i, 'alt_industry_description_2'] = pred.alternatives[1].description
            df.loc[i, 'alt_confidence_2'] = pred.alternatives[1].confidence
    
    # Create temporary file for response
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"bcea_predictions_{timestamp}.csv"
    
    # Use a directory that will be accessible
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, output_filename)
    
    # Save predictions to CSV
    df.to_csv(output_path, index=False)
    
    # Return CSV file as response
    return FileResponse(
        path=output_path,
        filename=output_filename,
        media_type='text/csv'
    )