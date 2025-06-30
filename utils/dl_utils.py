import logging
from typing import Any

import torch
from torch import nn

__all__ = [
    "load_pretrained_params",
    "_copy_tensor",
    "_bf16_to_float32",
    "_CompiledModule",
    "set_device_and_dtype"
]

def set_device_and_dtype(
    model: Any, batches: list[torch.Tensor], device: str | torch.device, dtype: torch.dtype
) -> tuple[Any, list[torch.Tensor]]:
    """Set the device and dtype of a model and its batches
    """
    return model.to(device=device, dtype=dtype), [batch.to(device=device, dtype=dtype) for batch in batches]

# torch compiled model type
_CompiledModule = torch._dynamo.eval_frame.OptimizedModule


def _copy_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.clone().detach()


def _bf16_to_float32(x: torch.Tensor) -> torch.Tensor:
    # bfloat16 is not supported in .numpy(): torch/csrc/utils/tensor_numpy.cpp:aten_to_numpy_dtype
    return x.float() if x.dtype == torch.bfloat16 else x


def load_pretrained_params(
    model: nn.Module,
    url: str | None = None,
    hash_prefix: str | None = None,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """Load a set of parameters onto a model

    >>> from receipt_cr.utils.dl_utils import load_pretrained_params
    >>> load_pretrained_params(model, "https://yoursource.com/yourcheckpoint-yourhash.zip")

    Args:
        model: the PyTorch model to be loaded
        url: URL of the zipped set of parameters
        hash_prefix: first characters of SHA256 expected hash
        ignore_keys: list of weights to be ignored from the state_dict
    """
    if url is None:
        logging.warning("Invalid model URL, using default initialization.")
    else:
        archive_path = download_from_url(url, hash_prefix=hash_prefix, cache_subdir="models", **kwargs)

        # Read state_dict
        state_dict = torch.load(archive_path, map_location="cpu")

        # Remove weights from the state_dict
        if ignore_keys is not None and len(ignore_keys) > 0:
            for key in ignore_keys:
                state_dict.pop(key)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if set(missing_keys) != set(ignore_keys) or len(unexpected_keys) > 0:
                raise ValueError("unable to load state_dict, due to non-matching keys.")
        else:
            # Load weights
            model.load_state_dict(state_dict)
