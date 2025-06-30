"""
Scene Text Recognition using PARSeq (Permuted Autoregressive Sequence Models)
Paper: https://arxiv.org/abs/2207.06966
Repo: https://github.com/baudm/parseq
"""

import math
from collections.abc import Callable
from copy import deepcopy
from itertools import permutations
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter


# Internal project imports (adjust if paths change)
from doctane.datasets import VOCABS
from doctane.models.recognition.utils import encode_sequences
from doctane.models.recognition.models.transformer_modules import MultiHeadAttention, PositionwiseFeedForward
from doctane.utils.dl_utils import _bf16_to_float32
from doctane.models.recognition.models.vitstr_classifier import vit_s
from doctane.models.recognition.recog_utils import RecognitionPostProcessor


__all__ = ["PARSeq", "parseq"]


default_cfgs: dict[str, dict[str, Any]] = {
    "parseq": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url/path": "",
    },
}


# ----------------------
# Abstract Target Handler
# ----------------------

class _PARSeq:
    vocab: str
    max_length: int

    def build_target(self, gts: list[str]) -> tuple[np.ndarray, list[int]]:
        """Encodes target strings into index sequences and returns their lengths."""
        encoded = encode_sequences(
            sequences=gts,
            vocab=self.vocab,
            target_size=self.max_length,
            eos=len(self.vocab),
            sos=len(self.vocab) + 1,
            pad=len(self.vocab) + 2,
        )
        seq_len = [len(word) for word in gts]
        return encoded, seq_len


class _PARSeqPostProcessor(RecognitionPostProcessor):
    """Post-processor converting model outputs into text predictions."""

    def __init__(self, vocab: str) -> None:
        super().__init__(vocab)
        self._embedding = list(vocab) + ["<eos>", "<sos>", "<pad>"]

# ----------------------
# Character Embedding
# ----------------------

class CharEmbedding(nn.Module):
    """Character embedding module for target tokens."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(self.d_model) * self.embedding(x)

# ----------------------
# PARSeq Decoder Block
# ----------------------

class PARSeqDecoder(nn.Module):
    """Transformer-style decoder used in PARSeq."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 12,
        ffd: int = 2048,
        ffd_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.cross_attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.position_feed_forward = PositionwiseFeedForward(d_model, ffd * ffd_ratio, dropout, nn.GELU())

        self.attention_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.cross_attention_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.query_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.content_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.feed_forward_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.output_norm = nn.LayerNorm(d_model, eps=1e-5)

        self.attention_dropout = nn.Dropout(dropout)
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(
        self,
        target,
        content,
        memory,
        target_mask: torch.Tensor | None = None,
    ):
        """Applies self-attention, cross-attention, and feed-forward."""
        query_norm = self.query_norm(target)
        content_norm = self.content_norm(content)

        target = target.clone() + self.attention_dropout(
            self.attention(query_norm, content_norm, content_norm, mask=target_mask)
        )
        target = target.clone() + self.cross_attention_dropout(
            self.cross_attention(self.query_norm(target), memory, memory)
        )
        target = target.clone() + self.feed_forward_dropout(
            self.position_feed_forward(self.feed_forward_norm(target))
        )
        return self.output_norm(target)

# ----------------------
# PARSeq Model
# ----------------------

class PARSeq(_PARSeq, nn.Module):
    """PARSeq model implementation."""

    def __init__(
        self,
        feature_extractor,
        vocab: str,
        embedding_units: int,
        max_length: int = 32,
        dropout_prob: float = 0.1,
        dec_num_heads: int = 12,
        dec_ff_dim: int = 384,
        dec_ffd_ratio: int = 4,
        input_shape: tuple[int, int, int] = (3, 32, 128),
        exportable: bool = False,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length
        self.vocab_size = len(vocab)
        self.rng = np.random.default_rng()

        # Network components
        self.feat_extractor = feature_extractor
        self.decoder = PARSeqDecoder(embedding_units, dec_num_heads, dec_ff_dim, dec_ffd_ratio, dropout_prob)
        self.head = nn.Linear(embedding_units, self.vocab_size + 1)  # EOS
        self.embed = CharEmbedding(self.vocab_size + 3, embedding_units)  # +3: SOS, EOS, PAD

        self.pos_queries = nn.Parameter(torch.Tensor(1, self.max_length + 1, embedding_units))  # +1 EOS
        self.dropout = nn.Dropout(p=dropout_prob)
        self.postprocessor = PARSeqPostProcessor(vocab=self.vocab)

        # Weight Initialization
        nn.init.trunc_normal_(self.pos_queries, std=0.02)
        for n, m in self.named_modules():
            if n.startswith("feat_extractor."):
                continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if getattr(m, "padding_idx", None) is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        target: list[str] | None = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> dict[str, Any]:
        """Forward pass of the model."""
        features = self.feat_extractor(x)["features"]
        features = features[:, 1:, :]  # Remove CLS token

        if self.training and target is None:
            raise ValueError("Need to provide labels during training")

        # Training mode with targets
        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt = torch.from_numpy(_gt).long().to(x.device)
            seq_len = torch.tensor(_seq_len).to(x.device)

            # Truncate targets for batching
            gt = gt[:, : int(seq_len.max().item()) + 2]
            if self.training:
                loss = self._training_step(gt, seq_len, features)
            else:
                gt = gt[:, 1:]  # remove SOS
                max_len = gt.shape[1] - 1
                logits = self.decode_autoregressive(features, max_len)
                loss = F.cross_entropy(logits.flatten(end_dim=1), gt.flatten(), ignore_index=self.vocab_size + 2)
        else:
            logits = self.decode_autoregressive(features)

        logits = _bf16_to_float32(logits)

        out: dict[str, Any] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output:
            out["out_map"] = logits

        if target is None or return_preds:
            @torch.compiler.disable  # Avoid errors with torch.compile
            def _postprocess(logits: torch.Tensor) -> list[tuple[str, float]]:
                return self.postprocessor(logits)

            out["preds"] = _postprocess(logits)

        if target is not None:
            out["loss"] = loss

        return out

    def generate_permutations(
        self, batch_size: int, seqlen: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates a batch of unique permutations for training."""
        max_perms = math.factorial(seqlen)
        num_perms = min(self.cfg.get("num_permutations", 6), max_perms)
        all_perms = list(permutations(range(seqlen)))
        perm_idxs = self.rng.choice(len(all_perms), size=num_perms, replace=False)

        perm_tensor = torch.tensor([all_perms[i] for i in perm_idxs], dtype=torch.long, device=device)
        perm_tensor = perm_tensor.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, P, L)
        inverse_perm = torch.argsort(perm_tensor, dim=2)

        return perm_tensor, inverse_perm

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        max_len: int,
        add_pos: bool = True,
    ) -> torch.Tensor:
        """Decodes using the decoder network with optional positional embeddings."""
        bs, length = tgt.size()
        tgt_embed = self.embed(tgt)

        if add_pos:
            pos_embed = self.pos_queries[:, 1:length + 1, :]
            tgt_embed = tgt_embed + pos_embed

        tgt_embed = self.dropout(tgt_embed)

        pos_queries = self.pos_queries[:, :max_len + 1, :]
        pos_queries = pos_queries.expand(bs, -1, -1)

        output = self.decoder(pos_queries[:, 1:], tgt_embed, memory)
        return self.head(output)

    def decode_autoregressive(
        self,
        memory: torch.Tensor,
        max_len: int | None = None,
    ) -> torch.Tensor:
        """Performs autoregressive decoding for inference."""
        bs, _, _ = memory.size()
        max_len = max_len or self.max_length

        preds = torch.full((bs, max_len), self.vocab_size + 2, dtype=torch.long, device=memory.device)  # PAD
        preds[:, 0] = self.vocab_size + 1  # SOS

        pos_queries = self.pos_queries[:, 1:max_len + 1, :].expand(bs, -1, -1)

        for i in range(max_len - 1):
            tgt = preds[:, :i + 1]
            tgt_embed = self.embed(tgt)
            tgt_embed = tgt_embed + pos_queries[:, :i + 1]
            tgt_embed = self.dropout(tgt_embed)

            output = self.decoder(pos_queries[:, :i + 1], tgt_embed, memory)
            logit = self.head(output[:, -1, :])
            preds[:, i + 1] = logit.argmax(-1)

        return self.head(self.decoder(pos_queries, self.embed(preds) + pos_queries, memory))

    def _training_step(
        self,
        tgt: torch.Tensor,
        tgt_len: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """Computes training loss using permuted sequences."""
        bs, seqlen = tgt.size()
        seqlen -= 1  # remove EOS

        device = tgt.device
        perms, inv_perms = self.generate_permutations(bs, seqlen, device)

        # Remove SOS (first token)
        tgt_input = tgt[:, :-1].unsqueeze(1).expand(-1, perms.size(1), -1)
        perms_tgt = torch.gather(tgt_input, 2, perms)

        tgt_embed = self.embed(perms_tgt)
        tgt_embed = tgt_embed + self.pos_queries[:, 1:seqlen + 1, :].unsqueeze(1)
        tgt_embed = self.dropout(tgt_embed)

        memory_exp = memory.unsqueeze(1).expand(-1, perms.size(1), -1, -1)

        tgt_embed = tgt_embed.reshape(-1, seqlen, tgt_embed.size(-1))
        memory_exp = memory_exp.reshape(-1, memory.size(1), memory.size(2))
        perms = perms.reshape(-1, seqlen)
        inv_perms = inv_perms.reshape(-1, seqlen)

        out = self.decoder(
            torch.gather(self.pos_queries[:, 1:seqlen + 1, :].expand(tgt_embed.size(0), -1, -1), 1, perms.unsqueeze(-1).expand(-1, -1, tgt_embed.size(-1))),
            tgt_embed,
            memory_exp,
        )
        out = self.head(out)

        out = torch.gather(out, 1, inv_perms.unsqueeze(-1).expand(-1, -1, out.size(-1)))

        tgt_output = tgt[:, 1:].unsqueeze(1).expand(-1, perms.size(1), -1).reshape(-1, seqlen)
        loss = F.cross_entropy(out.flatten(end_dim=1), tgt_output.flatten(), ignore_index=self.vocab_size + 2)

        return loss


class PARSeqPostProcessor(_PARSeqPostProcessor):
    """Post processor for PARSeq architecture

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __call__(
        self,
        logits: torch.Tensor,
    ) -> list[tuple[str, float]]:
        # compute pred with argmax for attention models
        out_idxs = logits.argmax(-1)
        preds_prob = torch.softmax(logits, -1).max(dim=-1)[0]

        # Manual decoding
        word_values = [
            "".join(self._embedding[idx] for idx in encoded_seq).split("<eos>")[0]
            for encoded_seq in out_idxs.cpu().numpy()
        ]
        # compute probabilties for each word up to the EOS token
        probs = [
            preds_prob[i, : len(word)].clip(0, 1).mean().item() if word else 0.0 for i, word in enumerate(word_values)
        ]

        return list(zip(word_values, probs))

def _parseq(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    layer: str,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> PARSeq:
    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])
    patch_size = kwargs.get("patch_size", (4, 8))

    kwargs["vocab"] = _cfg["vocab"]
    kwargs["input_shape"] = _cfg["input_shape"]

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        # NOTE: we don't use a pretrained backbone for non-rectangular patches to avoid the pos embed mismatch
        backbone_fn(False, input_shape=_cfg["input_shape"], patch_size=patch_size),  # type: ignore[call-arg]
        {layer: "features"},
    )

    kwargs.pop("patch_size", None)
    kwargs.pop("pretrained_backbone", None)

    # Build the model
    model = PARSeq(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if _cfg["vocab"] != default_cfgs[arch]["vocab"] else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def parseq(pretrained: bool = False, **kwargs: Any) -> PARSeq:
    """PARSeq architecture from
    `"Scene Text Recognition with Permuted Autoregressive Sequence Models" <https://arxiv.org/pdf/2207.06966>`_.

    >>> import torch
    >>> from receipt_cr.recognition.models import parseq
    >>> model = parseq(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 128))
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
        **kwargs: keyword arguments of the PARSeq architecture

    Returns:
        text recognition architecture
    """
    return _parseq(
        "parseq",
        pretrained,
        vit_s,
        "1",
        embedding_units=384,
        patch_size=(4, 8),
        ignore_keys=["embed.embedding.weight", "head.weight", "head.bias"],
        **kwargs,
    )
