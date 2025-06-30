import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from doctane.datasets import VOCABS
from doctane.models.recognition.utils import encode_sequences
from doctane.models.recognition.recog_utils import RecognitionPostProcessor
from doctane.models.recognition.models.vitstr_classifier import vit_t, vit_b, vit_s
from doctane.utils.dl_utils import _bf16_to_float32, load_pretrained_params

__all__ = ["ViTSTR", "vitstr_tiny", "vitstr_small", "vitstr_base"]

default_cfgs: dict[str, dict[str, Any]] = {
    "vitstr_tiny": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url/path": "",
    },
    "vitstr_small": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url/path": "",
    },
    "vitstr_base": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url/path": "",
    },
}


class ViTSTRPostProcessor(RecognitionPostProcessor):
    def __init__(self, vocab: str) -> None:
        super().__init__(vocab)
        self._embedding = list(vocab) + ["<eos>", "<sos>"]

    def __call__(self, logits: torch.Tensor) -> list[tuple[str, float]]:
        out_idxs = logits.argmax(-1)
        preds_prob = torch.softmax(logits, -1).max(dim=-1)[0]
        word_values = [
            "".join(self._embedding[idx] for idx in encoded_seq).split("<eos>")[0]
            for encoded_seq in out_idxs.cpu().numpy()
        ]
        probs = [
            preds_prob[i, : len(word)].clip(0, 1).mean().item() if word else 0.0 for i, word in enumerate(word_values)
        ]
        return list(zip(word_values, probs))


class ViTSTR(nn.Module):
    def __init__(
        self,
        feature_extractor,
        vocab: str,
        embedding_units: int,
        max_length: int = 32,
        input_shape: tuple[int, int, int] = (3, 32, 128),
        exportable: bool = False,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length + 2  # +2 for SOS and EOS

        self.feat_extractor = feature_extractor
        self.head = nn.Linear(embedding_units, len(self.vocab) + 1)  # +1 for EOS
        self.postprocessor = ViTSTRPostProcessor(vocab=self.vocab)

    def build_target(
        self,
        gts: list[str],
    ) -> tuple[np.ndarray, list[int]]:
        encoded = encode_sequences(
            sequences=gts,
            vocab=self.vocab,
            target_size=self.max_length,
            eos=len(self.vocab),
            sos=len(self.vocab) + 1,
        )
        seq_len = [len(word) for word in gts]
        return encoded, seq_len

    def forward(
        self,
        x: torch.Tensor,
        target: list[str] | None = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> dict[str, Any]:
        features = self.feat_extractor(x)["features"]
        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long), torch.tensor(_seq_len)
            gt, seq_len = gt.to(x.device), seq_len.to(x.device)

        if self.training and target is None:
            raise ValueError("Need to provide labels during training")

        features = features[:, : self.max_length]
        B, N, E = features.size()
        features = features.reshape(B * N, E)
        logits = self.head(features).view(B, N, len(self.vocab) + 1)
        decoded_features = _bf16_to_float32(logits[:, 1:])

        out: dict[str, Any] = {}
        if self.exportable:
            out["logits"] = decoded_features
            return out

        if return_model_output:
            out["out_map"] = decoded_features

        if target is None or return_preds:
            @torch.compiler.disable
            def _postprocess(decoded_features: torch.Tensor) -> list[tuple[str, float]]:
                return self.postprocessor(decoded_features)

            out["preds"] = _postprocess(decoded_features)

        if target is not None:
            out["loss"] = self.compute_loss(decoded_features, gt, seq_len)

        return out

    @staticmethod
    def compute_loss(
        model_output: torch.Tensor,
        gt: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        input_len = model_output.shape[1]
        seq_len = seq_len + 1
        cce = F.cross_entropy(model_output.permute(0, 2, 1), gt[:, 1:], reduction="none")
        mask_2d = torch.arange(input_len, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0
        ce_loss = cce.sum(1) / seq_len.to(dtype=model_output.dtype)
        return ce_loss.mean()


def _vitstr(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    layer: str,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> ViTSTR:
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])
    patch_size = kwargs.get("patch_size", (4, 8))

    kwargs["vocab"] = _cfg["vocab"]
    kwargs["input_shape"] = _cfg["input_shape"]

    feat_extractor = IntermediateLayerGetter(
        backbone_fn(False, input_shape=_cfg["input_shape"], patch_size=patch_size),
        {layer: "features"},
    )

    kwargs.pop("patch_size", None)
    kwargs.pop("pretrained_backbone", None)

    model = ViTSTR(feat_extractor, cfg=_cfg, **kwargs)
    if pretrained:
        _ignore_keys = ignore_keys if _cfg["vocab"] != default_cfgs[arch]["vocab"] else None
        load_pretrained_params(model, default_cfgs[arch]["url/path"], ignore_keys=_ignore_keys)

    return model


def vitstr_tiny(pretrained: bool = False, **kwargs: Any) -> ViTSTR:
    return _vitstr(
        "vitstr_tiny",
        pretrained,
        vit_t,
        "1",
        embedding_units=192,
        patch_size=(4, 8),
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )


def vitstr_small(pretrained: bool = False, **kwargs: Any) -> ViTSTR:
    return _vitstr(
        "vitstr_small",
        pretrained,
        vit_s,
        "1",
        embedding_units=384,
        patch_size=(4, 8),
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )


def vitstr_base(pretrained: bool = False, **kwargs: Any) -> ViTSTR:
    return _vitstr(
        "vitstr_base",
        pretrained,
        vit_b,
        "1",
        embedding_units=768,
        patch_size=(4, 8),
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )
