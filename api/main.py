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


# ============= Configuration =============

API_VERSION = "1.0.0"
APP_NAME = "Doctane API"

# Global predictor instance (lazy loaded)
_predictor: Optional[Any] = None
_model_config = {
    "detection": "seg_linknet_resnet50",
    "recognition": "sar_resnet34"
}

# ============= Available Models =============

DETECTION_MODELS = {
    # LinkNet models
    "seg_linknet_resnet50": "LinkNet with ResNet50 backbone",
    "seg_linknet_resnet101": "LinkNet with ResNet101 backbone",
    "seg_linknet_resnet18": "LinkNet with ResNet18 backbone",
    "seg_linknet_resnet34": "LinkNet with ResNet34 backbone",
    "seg_linknet_efficientnet_b0": "LinkNet with EfficientNet-B0",
    "seg_linknet_efficientnet_b1": "LinkNet with EfficientNet-B1",
    "seg_linknet_efficientnet_b2": "LinkNet with EfficientNet-B2",
    "seg_linknet_efficientnet_b3": "LinkNet with EfficientNet-B3",
    "seg_linknet_efficientnet_b4": "LinkNet with EfficientNet-B4",
    "seg_linknet_efficientnet_b5": "LinkNet with EfficientNet-B5",
    "seg_linknet_mobilenet_v2": "LinkNet with MobileNetV2",
    "seg_linknet_mit_b0": "LinkNet with MiT-B0 encoder",
    "seg_linknet_mit_b1": "LinkNet with MiT-B1 encoder",
    "seg_linknet_mit_b2": "LinkNet with MiT-B2 encoder",
    "seg_linknet_mit_b3": "LinkNet with MiT-B3 encoder",
    "seg_linknet_mit_b4": "LinkNet with MiT-B4 encoder",
    "seg_linknet_mit_b5": "LinkNet with MiT-B5 encoder",
    
    # DeepLabV3+ models
    "seg_deeplabv3plus_resnet50": "DeepLabV3+ with ResNet50 backbone",
    "seg_deeplabv3plus_resnet101": "DeepLabV3+ with ResNet101 backbone",
    "seg_deeplabv3plus_resnet18": "DeepLabV3+ with ResNet18 backbone",
    "seg_deeplabv3plus_resnet34": "DeepLabV3+ with ResNet34 backbone",
    "seg_deeplabv3plus_efficientnet_b0": "DeepLabV3+ with EfficientNet-B0",
    "seg_deeplabv3plus_efficientnet_b1": "DeepLabV3+ with EfficientNet-B1",
    "seg_deeplabv3plus_efficientnet_b2": "DeepLabV3+ with EfficientNet-B2",
    "seg_deeplabv3plus_mobilenet_v2": "DeepLabV3+ with MobileNetV2",
    
    # SegFormer models
    "seg_segformer_resnet50": "SegFormer with ResNet50 backbone",
    "seg_segformer_resnet101": "SegFormer with ResNet101 backbone",
    "seg_segformer_mit_b0": "SegFormer with MiT-B0 encoder",
    "seg_segformer_mit_b1": "SegFormer with MiT-B1 encoder",
    "seg_segformer_mit_b2": "SegFormer with MiT-B2 encoder",
    "seg_segformer_mit_b3": "SegFormer with MiT-B3 encoder",
    "seg_segformer_mit_b4": "SegFormer with MiT-B4 encoder",
    "seg_segformer_mit_b5": "SegFormer with MiT-B5 encoder",
    
    # UNet models (most popular encoders)
    "seg_unet_resnet18": "UNet with ResNet18 encoder",
    "seg_unet_resnet34": "UNet with ResNet34 encoder",
    "seg_unet_resnet50": "UNet with ResNet50 encoder",
    "seg_unet_resnet101": "UNet with ResNet101 encoder",
    "seg_unet_efficientnet_b0": "UNet with EfficientNet-B0",
    "seg_unet_efficientnet_b1": "UNet with EfficientNet-B1",
    "seg_unet_efficientnet_b2": "UNet with EfficientNet-B2",
    "seg_unet_efficientnet_b3": "UNet with EfficientNet-B3",
    "seg_unet_efficientnet_b4": "UNet with EfficientNet-B4",
    "seg_unet_efficientnet_b5": "UNet with EfficientNet-B5",
    "seg_unet_efficientnet_b6": "UNet with EfficientNet-B6",
    "seg_unet_efficientnet_b7": "UNet with EfficientNet-B7",
    "seg_unet_mobilenet_v2": "UNet with MobileNetV2",
    "seg_unet_vgg11": "UNet with VGG11 encoder",
    "seg_unet_vgg11_bn": "UNet with VGG11-BN encoder",
    "seg_unet_vgg13": "UNet with VGG13 encoder",
    "seg_unet_vgg13_bn": "UNet with VGG13-BN encoder",
    "seg_unet_vgg16": "UNet with VGG16 encoder",
    "seg_unet_vgg16_bn": "UNet with VGG16-BN encoder",
    "seg_unet_vgg19": "UNet with VGG19 encoder",
    "seg_unet_vgg19_bn": "UNet with VGG19-BN encoder",
    "seg_unet_densenet121": "UNet with DenseNet121",
    "seg_unet_densenet169": "UNet with DenseNet169",
    "seg_unet_densenet201": "UNet with DenseNet201",
    "seg_unet_densenet161": "UNet with DenseNet161",
    "seg_unet_se_resnet50": "UNet with SE-ResNet50",
    "seg_unet_se_resnet101": "UNet with SE-ResNet101",
    "seg_unet_se_resnet152": "UNet with SE-ResNet152",
    "seg_unet_se_resnext50_32x4d": "UNet with SE-ResNeXt50",
    "seg_unet_se_resnext101_32x4d": "UNet with SE-ResNeXt101",
    "seg_unet_timm_efficientnet_b0": "UNet with TIMM EfficientNet-B0",
    "seg_unet_timm_efficientnet_b1": "UNet with TIMM EfficientNet-B1",
    "seg_unet_timm_efficientnet_b2": "UNet with TIMM EfficientNet-B2",
    "seg_unet_timm_efficientnet_b3": "UNet with TIMM EfficientNet-B3",
    "seg_unet_timm_efficientnet_b4": "UNet with TIMM EfficientNet-B4",
    "seg_unet_timm_efficientnet_b5": "UNet with TIMM EfficientNet-B5",
    "seg_unet_timm_efficientnet_b6": "UNet with TIMM EfficientNet-B6",
    "seg_unet_timm_efficientnet_b7": "UNet with TIMM EfficientNet-B7",
    "seg_unet_mit_b0": "UNet with MiT-B0 encoder",
    "seg_unet_mit_b1": "UNet with MiT-B1 encoder",
    "seg_unet_mit_b2": "UNet with MiT-B2 encoder",
    "seg_unet_mit_b3": "UNet with MiT-B3 encoder",
    "seg_unet_mit_b4": "UNet with MiT-B4 encoder",
    "seg_unet_mit_b5": "UNet with MiT-B5 encoder",
    
    # UNet++ models
    "seg_unetplusplus_resnet18": "UNet++ with ResNet18 encoder",
    "seg_unetplusplus_resnet34": "UNet++ with ResNet34 encoder",
    "seg_unetplusplus_resnet50": "UNet++ with ResNet50 encoder",
    "seg_unetplusplus_resnet101": "UNet++ with ResNet101 encoder",
    "seg_unetplusplus_efficientnet_b0": "UNet++ with EfficientNet-B0",
    "seg_unetplusplus_efficientnet_b1": "UNet++ with EfficientNet-B1",
    "seg_unetplusplus_efficientnet_b2": "UNet++ with EfficientNet-B2",
    "seg_unetplusplus_efficientnet_b3": "UNet++ with EfficientNet-B3",
    "seg_unetplusplus_efficientnet_b4": "UNet++ with EfficientNet-B4",
    "seg_unetplusplus_mobilenet_v2": "UNet++ with MobileNetV2",
    "seg_unetplusplus_mit_b0": "UNet++ with MiT-B0 encoder",
    "seg_unetplusplus_mit_b1": "UNet++ with MiT-B1 encoder",
    "seg_unetplusplus_mit_b2": "UNet++ with MiT-B2 encoder",
    "seg_unetplusplus_mit_b3": "UNet++ with MiT-B3 encoder",
    "seg_unetplusplus_mit_b4": "UNet++ with MiT-B4 encoder",
    "seg_unetplusplus_mit_b5": "UNet++ with MiT-B5 encoder",
    
    # FPN models
    "seg_fpn_resnet18": "FPN with ResNet18 encoder",
    "seg_fpn_resnet34": "FPN with ResNet34 encoder",
    "seg_fpn_resnet50": "FPN with ResNet50 encoder",
    "seg_fpn_resnet101": "FPN with ResNet101 encoder",
    "seg_fpn_efficientnet_b0": "FPN with EfficientNet-B0",
    "seg_fpn_efficientnet_b1": "FPN with EfficientNet-B1",
    "seg_fpn_efficientnet_b2": "FPN with EfficientNet-B2",
    "seg_fpn_mobilenet_v2": "FPN with MobileNetV2",
    "seg_fpn_mit_b0": "FPN with MiT-B0 encoder",
    "seg_fpn_mit_b1": "FPN with MiT-B1 encoder",
    "seg_fpn_mit_b2": "FPN with MiT-B2 encoder",
    "seg_fpn_mit_b3": "FPN with MiT-B3 encoder",
    "seg_fpn_mit_b4": "FPN with MiT-B4 encoder",
    
    # PSPNet models
    "seg_pspnet_resnet18": "PSPNet with ResNet18 encoder",
    "seg_pspnet_resnet34": "PSPNet with ResNet34 encoder",
    "seg_pspnet_resnet50": "PSPNet with ResNet50 encoder",
    "seg_pspnet_resnet101": "PSPNet with ResNet101 encoder",
    "seg_pspnet_efficientnet_b0": "PSPNet with EfficientNet-B0",
    "seg_pspnet_efficientnet_b1": "PSPNet with EfficientNet-B1",
    "seg_pspnet_efficientnet_b2": "PSPNet with EfficientNet-B2",
    "seg_pspnet_mobilenet_v2": "PSPNet with MobileNetV2",
    "seg_pspnet_mit_b0": "PSPNet with MiT-B0 encoder",
    "seg_pspnet_mit_b1": "PSPNet with MiT-B1 encoder",
    "seg_pspnet_mit_b2": "PSPNet with MiT-B2 encoder",
    "seg_pspnet_mit_b3": "PSPNet with MiT-B3 encoder",
    "seg_pspnet_mit_b4": "PSPNet with MiT-B4 encoder",
    
    # PAN (Path Aggregation Network)
    "seg_pan_resnet18": "PAN with ResNet18 encoder",
    "seg_pan_resnet34": "PAN with ResNet34 encoder",
    "seg_pan_resnet50": "PAN with ResNet50 encoder",
    "seg_pan_resnet101": "PAN with ResNet101 encoder",
    "seg_pan_efficientnet_b0": "PAN with EfficientNet-B0",
    "seg_pan_efficientnet_b1": "PAN with EfficientNet-B1",
    "seg_pan_efficientnet_b2": "PAN with EfficientNet-B2",
    "seg_pan_mobilenet_v2": "PAN with MobileNetV2",
    "seg_pan_mit_b0": "PAN with MiT-B0 encoder",
    "seg_pan_mit_b1": "PAN with MiT-B1 encoder",
    "seg_pan_mit_b2": "PAN with MiT-B2 encoder",
    "seg_pan_mit_b3": "PAN with MiT-B3 encoder",
    "seg_pan_mit_b4": "PAN with MiT-B4 encoder",
}

RECOGNITION_MODELS = {
    # SAR models
    "sar_resnet18": "SAR with ResNet18 backbone",
    "sar_resnet34": "SAR with ResNet34 backbone",
    "sar_resnet50": "SAR with ResNet50 backbone",
    "sar_resnet101": "SAR with ResNet101 backbone",
    
    # ViTSTR models
    "vitstr_tiny": "ViTSTR Tiny",
    "vitstr_small": "ViTSTR Small",
    "vitstr_base": "ViTSTR Base",
    
    # CRNN models
    "crnn_resnet18": "CRNN with ResNet18",
    "crnn_resnet34": "CRNN with ResNet34",
    "crnn_resnet50": "CRNN with ResNet50",
    "crnn_mobilenet_v2": "CRNN with MobileNetV2",
    
    # MASTER models
    "master_resnet18": "MASTER with ResNet18",
    "master_resnet34": "MASTER with ResNet34",
    "master_resnet50": "MASTER with ResNet50",
    
    # TRBA models
    "trba_resnet18": "TRBA with ResNet18",
    "trba_resnet34": "TRBA with ResNet34",
    "trba_resnet50": "TRBA with ResNet50",
    
    # ABINet models
    "abinet_resnet18": "ABINet with ResNet18",
    "abinet_resnet34": "ABINet with ResNet34",
    
    # LSTR models
    "lstr_resnet18": "LSTR with ResNet18",
    "lstr_resnet34": "LSTR with ResNet34",
    
    # ViTPTR models
    "viptr_resnet18": "ViTPTR with ResNet18",
    "viptr_resnet34": "ViTPTR with ResNet34",
    
    # MATRN models
    "matrn_resnet18": "MATRN with ResNet18",
    "matrn_resnet34": "MATRN with ResNet34",
    
    # PARSeq models
    "parseq_tiny": "PARSeq Tiny",
    "parseq_small": "PARSeq Small",
    "parseq_base": "PARSeq Base",
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
    """Load detection model by name dynamically"""
    from doctane.models.detection import smp_model_zoo as detection_zoo
    
    if model_name not in DETECTION_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown detection model: {model_name}")
    
    model_func = getattr(detection_zoo, model_name, None)
    if model_func is None:
        raise HTTPException(status_code=400, detail=f"Detection model not found: {model_name}")
    
    return model_func


def get_recognition_model(model_name: str):
    """Load recognition model by name dynamically"""
    from doctane.models.recognition import models as recognition_zoo
    
    if model_name not in RECOGNITION_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown recognition model: {model_name}")
    
    model_func = getattr(recognition_zoo, model_name, None)
    if model_func is None:
        raise HTTPException(status_code=400, detail=f"Recognition model not found: {model_name}")
    
    return model_func


def get_predictor(det_model: str, reco_model: str, **kwargs):
    """Get or create OCR predictor with specified models"""
    global _predictor, _model_config
    
    try:
        from doctane.ocr_pipeline.ocr_predictor import OCRPredictor
    except ImportError:
        return None
    
    if (_predictor is None or 
        _model_config["detection"] != det_model or 
        _model_config["recognition"] != reco_model):
        
        try:
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
        except Exception as e:
            print(f"Warning: Could not load models: {e}")
            _predictor = None
    
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
            {"name": name, "type": "detection", "description": desc}
            for name, desc in DETECTION_MODELS.items()
        ],
        "recognition": [
            {"name": name, "type": "recognition", "description": desc}
            for name, desc in RECOGNITION_MODELS.items()
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