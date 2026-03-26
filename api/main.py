"""
Doctane API - FastAPI Backend for Document OCR

Provides RESTful endpoints for document processing, model management, and OCR operations.
"""

import os
import io
import base64
import json
from typing import Any, Optional
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import torch
import uvicorn

from doctane.ocr_pipeline.ocr_predictor import OCRPredictor
from doctane.models.detection.smp_model_zoo import (
    seg_linknet_resnet50,
    seg_deeplabv3plus_resnet50,
    seg_segformer_resnet50
)
from doctane.models.recognition.models import sar_resnet34, sar_resnet18


# ============= Configuration =============

API_VERSION = "1.0.0"
APP_NAME = "Doctane API"

# Global predictor instance (lazy loaded)
_predictor: Optional[OCRPredictor] = None
_model_config = {
    "detection": "seg_linknet_resnet50",
    "recognition": "sar_resnet34"
}

# ============= Pydantic Models =============

class ProcessRequest(BaseModel):
    """Request model for OCR processing"""
    model_detection: str = Field(default="seg_linknet_resnet50", description="Detection model name")
    model_recognition: str = Field(default="sar_resnet34", description="Recognition model name")
    assume_straight_pages: bool = Field(default=True, description="Assume pages are straight")
    detect_language: bool = Field(default=True, description="Detect language")


class OCRResult(BaseModel):
    """OCR processing result"""
    success: bool
    pages: list[dict[str, Any]]
    stats: dict[str, int]
    model_info: dict[str, str]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: bool


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    type: str
    description: str


# ============= FastAPI App =============

app = FastAPI(
    title="Doctane API",
    description="Intelligent Document Analysis & Understanding System",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============= Utility Functions =============

def get_detection_model(model_name: str):
    """Load detection model by name"""
    models = {
        "seg_linknet_resnet50": seg_linknet_resnet50,
        "seg_deeplabv3plus_resnet50": seg_deeplabv3plus_resnet50,
        "seg_segformer_resnet50": seg_segformer_resnet50,
    }
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Unknown detection model: {model_name}")
    return models[model_name]


def get_recognition_model(model_name: str):
    """Load recognition model by name"""
    models = {
        "sar_resnet34": sar_resnet34,
        "sar_resnet18": sar_resnet18,
    }
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Unknown recognition model: {model_name}")
    return models[model_name]


def get_predictor(det_model: str, reco_model: str, **kwargs) -> OCRPredictor:
    """Get or create OCR predictor with specified models"""
    global _predictor, _model_config
    
    if (_predictor is None or 
        _model_config["detection"] != det_model or 
        _model_config["recognition"] != reco_model):
        
        # Load models
        det_loader = get_detection_model(det_model)
        reco_loader = get_recognition_model(reco_model)
        
        det = det_loader(pretrained=False)
        reco = reco_loader(pretrained=False, pretrained_backbone=False)
        
        _predictor = OCRPredictor(
            det_predictor=det,
            reco_predictor=reco,
            **kwargs
        )
        _model_config = {"detection": det_model, "recognition": reco_model}
    
    return _predictor


def serialize_page(page) -> dict[str, Any]:
    """Serialize Page object to dictionary"""
    result = {}
    for key in dir(page):
        if key.startswith('_'):
            continue
        try:
            value = getattr(page, key)
            if hasattr(value, '__dict__'):
                result[key] = serialize_page(value)
            elif isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                result[key] = value
            elif hasattr(value, 'tolist'):
                result[key] = value.tolist()
        except:
            pass
    return result


def calculate_stats(pages: list) -> dict[str, int]:
    """Calculate statistics from OCR result"""
    stats = {
        "total_pages": len(pages),
        "total_words": 0,
        "total_lines": 0,
        "total_blocks": 0
    }
    
    for page in pages:
        words = 0
        lines = 0
        blocks = 0
        
        if hasattr(page, 'words'):
            words = len(page.words) if hasattr(page.words, '__len__') else 0
        if hasattr(page, 'lines'):
            lines = len(page.lines) if hasattr(page.lines, '__len__') else 0
        if hasattr(page, 'blocks'):
            blocks = len(page.blocks) if hasattr(page.blocks, '__len__') else 0
            
        stats["total_words"] += words
        stats["total_lines"] += lines
        stats["total_blocks"] += blocks
    
    return stats


# ============= API Endpoints =============

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": APP_NAME,
        "version": API_VERSION,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        models_loaded=_predictor is not None
    )


@app.get("/models", tags=["Models"])
async def list_models():
    """List available models"""
    return {
        "detection": [
            {"name": "seg_linknet_resnet50", "type": "detection", "description": "LinkNet with ResNet50 backbone"},
            {"name": "seg_deeplabv3plus_resnet50", "type": "detection", "description": "DeepLabV3+ with ResNet50"},
            {"name": "seg_segformer_resnet50", "type": "detection", "description": "SegFormer with ResNet50"},
        ],
        "recognition": [
            {"name": "sar_resnet34", "type": "recognition", "description": "SAR with ResNet34"},
            {"name": "sar_resnet18", "type": "recognition", "description": "SAR with ResNet18"},
        ]
    }


@app.post("/process", response_model=OCRResult, tags=["OCR"])
async def process_document(
    file: UploadFile = File(...),
    model_detection: str = Form(default="seg_linknet_resnet50"),
    model_recognition: str = Form(default="sar_resnet34"),
    assume_straight_pages: bool = Form(default=True),
    detect_language: bool = Form(default=True),
):
    """
    Process a document image through OCR pipeline.
    
    Upload an image file to extract text with detection and recognition models.
    """
    # Validate file type
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: PNG, JPEG, BMP, TIFF"
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        np_image = np.array(image)
        
        # Get predictor
        predictor = get_predictor(
            model_detection,
            model_recognition,
            assume_straight_pages=assume_straight_pages,
            detect_language=detect_language
        )
        
        # Process
        result = predictor([np_image])
        
        # Serialize results
        pages_data = []
        if hasattr(result, 'pages'):
            for page in result.pages:
                pages_data.append(serialize_page(page))
        
        stats = calculate_stats(pages_data)
        
        return OCRResult(
            success=True,
            pages=pages_data,
            stats=stats,
            model_info={
                "detection": model_detection,
                "recognition": model_recognition
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/base64", response_model=OCRResult, tags=["OCR"])
async def process_document_base64(
    image_data: str = Form(..., description="Base64 encoded image"),
    model_detection: str = Form(default="seg_linknet_resnet50"),
    model_recognition: str = Form(default="sar_resnet34"),
    assume_straight_pages: bool = Form(default=True),
):
    """
    Process a base64 encoded image through OCR pipeline.
    """
    try:
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(image)
        
        # Get predictor
        predictor = get_predictor(model_detection, model_recognition)
        
        # Process
        result = predictor([np_image])
        
        # Serialize results
        pages_data = []
        if hasattr(result, 'pages'):
            for page in result.pages:
                pages_data.append(serialize_page(page))
        
        return OCRResult(
            success=True,
            pages=pages_data,
            stats=calculate_stats(pages_data),
            model_info={"detection": model_detection, "recognition": model_recognition}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=dict, tags=["OCR"])
async def process_batch(
    files: list[UploadFile] = File(...),
    model_detection: str = Form(default="seg_linknet_resnet50"),
    model_recognition: str = Form(default="sar_resnet34"),
):
    """
    Process multiple document images in batch.
    """
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            np_image = np.array(image)
            
            predictor = get_predictor(model_detection, model_recognition)
            result = predictor([np_image])
            
            pages_data = []
            if hasattr(result, 'pages'):
                for page in result.pages:
                    pages_data.append(serialize_page(page))
            
            results.append({
                "filename": file.filename,
                "success": True,
                "pages": pages_data,
                "stats": calculate_stats(pages_data)
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"batch_size": len(files), "results": results}


# ============= Static Files =============

@app.get("/app", tags=["Frontend"])
async def serve_app():
    """Serve the frontend application"""
    app_path = Path(__file__).parent.parent / "app.html"
    return FileResponse(app_path)


# ============= Main =============

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting {APP_NAME} v{API_VERSION}")
    print(f"Server running on http://{host}:{port}")
    print(f"API docs available at http://{host}:{port}/docs")
    print(f"Frontend available at http://{host}:{port}/app")
    
    uvicorn.run(app, host=host, port=port)