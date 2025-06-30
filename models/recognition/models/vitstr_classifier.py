from copy import deepcopy
from typing import Any

import torch
from torch import nn

from doctane.datasets import VOCABS
from doctane.models.recognition.models.transformer_modules import EncoderBlock, PatchEmbedding

__all__ = ["vit_t", "vit_s", "vit_b"]

default_cfgs: dict[str, dict[str, Any]] = {
    "vit_t": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url/path": "",  # Provide pretrained URL/path here if available
    },
    "vit_s": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url/path": "",
    },
    "vit_b": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url/path": "",
    },
}


class ClassifierHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x[:, 0])


class VisionTransformer(nn.Sequential):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffd_ratio: int,
        patch_size: tuple[int, int] = (4, 4),
        input_shape: tuple[int, int, int] = (3, 32, 32),
        dropout: float = 0.0,
        num_classes: int = 1000,
        include_top: bool = True,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        _layers: list[nn.Module] = [
            PatchEmbedding(input_shape, d_model, patch_size),
            EncoderBlock(num_layers, num_heads, d_model, d_model * ffd_ratio, dropout, nn.GELU()),
        ]
        if include_top:
            _layers.append(ClassifierHead(d_model, num_classes))

        super().__init__(*_layers)
        self.cfg = cfg


def _vit(
    arch: str,
    pretrained: bool,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> VisionTransformer:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["input_shape"] = kwargs["input_shape"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    model = VisionTransformer(cfg=_cfg, **kwargs)
    if pretrained:
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        load_pretrained_params(model, default_cfgs[arch]["url/path"], ignore_keys=_ignore_keys)

    return model


def vit_t(pretrained: bool = False, **kwargs: Any) -> VisionTransformer:
    """VisionTransformer-Tiny architecture (ViT-T)

    >>> from receipt_cr.detection.models.vitstr_classifier import vit_t
    >>> model = vit_t(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 32))
    >>> out = model(input_tensor)

    Returns:
        A VisionTransformer-Tiny model
    """
    return _vit(
        "vit_t",
        pretrained,
        d_model=192,
        num_layers=12,
        num_heads=3,
        ffd_ratio=4,
        ignore_keys=["2.head.weight", "2.head.bias"],
        **kwargs,
    )


def vit_s(pretrained: bool = False, **kwargs: Any) -> VisionTransformer:
    return _vit(
        "vit_s",
        pretrained,
        d_model=384,
        num_layers=12,
        num_heads=6,
        ffd_ratio=4,
        ignore_keys=["2.head.weight", "2.head.bias"],
        **kwargs,
    )


def vit_b(pretrained: bool = False, **kwargs: Any) -> VisionTransformer:
    return _vit(
        "vit_b",
        pretrained,
        d_model=768,
        num_layers=12,
        num_heads=12,
        ffd_ratio=4,
        ignore_keys=["2.head.weight", "2.head.bias"],
        **kwargs,
    )
