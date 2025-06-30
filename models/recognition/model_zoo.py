from typing import Any

from doctane.models.utils.pre_processor import PreProcessor
from doctane.models.recognition import models as recognition
from doctane.models.recognition.recognition_predictor import RecognitionPredictor

__all__ = ["recognition_predictor"]

# Supported model architecture names for string-based lookup
ARCHS: list[str] = [
    "sar_resnet18",
    "sar_resnet34",
    "sar_resnet50",
    "sar_resnet101",
    "vitstr_tiny",
    "vitstr_small",
    "vitstr_base",
]

def _predictor(arch: Any, pretrained: bool, **kwargs: Any) -> RecognitionPredictor:
    """
    Internal function to instantiate a recognition predictor based on the given architecture.

    Args:
        arch: Either a string referring to a model name or a model class instance.
        pretrained: Whether to load pretrained weights.
        **kwargs: Optional keyword arguments for the model and preprocessing.

    Returns:
        A configured RecognitionPredictor instance.
    """
    # Handle string-based architecture lookup
    if isinstance(arch, str):
        if arch not in ARCHS:
            raise ValueError(f"Unknown architecture '{arch}'. Supported architectures are: {ARCHS}")
        
        _model = recognition.__dict__[arch](
            pretrained=pretrained,
            pretrained_backbone=kwargs.get("pretrained_backbone", True)
        )
    else:
        # Handle direct model object cases (e.g., SAR, ViTSTR, or compiled models)
        allowed_archs = [recognition.SAR, recognition.ViTSTR]

        if is_torch_available():
            # Extend for torch-only architectures like VIPTR
            allowed_archs.append(recognition.VIPTR)
            from doctr.models.utils import _CompiledModule
            allowed_archs.append(_CompiledModule)

        if not isinstance(arch, tuple(allowed_archs)):
            raise ValueError(f"Unsupported model type provided: {type(arch)}")
        
        _model = arch

    # Clean up optional kwargs
    kwargs.pop("pretrained_backbone", None)

    # Apply model configuration defaults if not provided
    kwargs["mean"] = kwargs.get("mean", _model.cfg["mean"])
    kwargs["std"] = kwargs.get("std", _model.cfg["std"])
    kwargs["batch_size"] = kwargs.get("batch_size", 128)

    # Determine input shape based on available backend (TensorFlow or Torch)
    input_shape = _model.cfg["input_shape"][:2] if is_tf_available() else _model.cfg["input_shape"][-2:]

    # Create and return the predictor instance
    return RecognitionPredictor(
        PreProcessor(input_shape, preserve_aspect_ratio=True, **kwargs),
        _model
    )

def recognition_predictor(
    arch: Any = "sar_resnet18",
    pretrained: bool = False,
    symmetric_pad: bool = False,
    batch_size: int = 128,
    **kwargs: Any,
) -> RecognitionPredictor:
    """
    Factory function to instantiate a text recognition model.

    Example:
        >>> import numpy as np
        >>> from doctane.models.recognition.model_zoo import recognition_predictor
        >>> model = recognition_predictor(pretrained=True)
        >>> input_page = (255 * np.random.rand(32, 128, 3)).astype(np.uint8)
        >>> out = model([input_page])

    Args:
        arch: Name of the architecture (string) or model instance.
        pretrained: Load pretrained weights if True.
        symmetric_pad: Apply symmetric padding if True.
        batch_size: Number of samples to process in parallel.
        **kwargs: Additional keyword arguments passed to PreProcessor.

    Returns:
        An instance of RecognitionPredictor.
    """
    return _predictor(
        arch=arch,
        pretrained=pretrained,
        symmetric_pad=symmetric_pad,
        batch_size=batch_size,
        **kwargs,
    )
