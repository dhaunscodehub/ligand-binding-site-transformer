"""Multimodal cross-attention transformer for per-residue binding prediction.

Architecture, and the reason for each piece:

**Two encoders.** Sequence features (identity + physicochemistry) and
structure features (backbone geometry) are embedded separately. Concatenating
them into one stream would let the first linear layer mix them immediately,
which makes it impossible to ask what each modality contributes — and the
ablation in :mod:`bindsite.validation` needs exactly that question answered.

**Cross-attention between the modalities.** Each stream attends to the other,
so a residue's structural context can be interpreted in light of its chemistry
and vice versa. This is the operation that makes the model "structure-aware"
rather than a sequence model with extra columns: a hydrophobic residue in a
concave pocket and the same residue on a convex surface get different
representations.

**Self-attention within each stream.** A binding site is a set of residues
that are close in space but often far apart in sequence. Self-attention over
the whole chain lets a residue see those partners; a convolutional or windowed
model cannot, which is the main architectural argument for attention here.

**Structural attention bias.** Raw self-attention has no notion of distance,
so it would treat a residue 3 Å away and one 40 Å away identically. A learned
bias derived from the CA-CA distance matrix is added to the attention logits,
which injects the geometry that makes a pocket a pocket. Implemented in
:class:`DistanceBias`.

**Per-residue output.** One logit per residue, trained with a class-weighted
loss because binding residues are a small minority of every protein.

Everything here is written against PyTorch's public API and runs on CPU, CUDA
or Apple MPS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..exceptions import ConfigError, OptionalDependencyMissing


def _torch():
    try:
        import torch
    except ImportError as error:
        raise OptionalDependencyMissing(
            "PyTorch is required for the transformer: pip install torch. "
            "The scikit-learn baselines in bindsite.models.baselines run "
            "without it."
        ) from error
    return torch


@dataclass
class TransformerConfig:
    """Architecture and training hyperparameters."""

    sequence_dim: int = 30
    structure_dim: int = 8
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ff_multiplier: int = 4
    dropout: float = 0.1
    max_length: int = 512
    use_distance_bias: bool = True
    # Which modalities are active. Both false is rejected; one false is the
    # ablation the validation suite runs.
    use_sequence: bool = True
    use_structure: bool = True

    def __post_init__(self) -> None:
        # Positivity first: n_heads is used as a divisor below, so a zero here
        # would raise ZeroDivisionError instead of a usable message.
        for name in ("d_model", "n_heads", "n_layers", "max_length"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")
        if self.d_model % self.n_heads:
            raise ConfigError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads}); each head takes d_model // n_heads channels"
            )
        if not (self.use_sequence or self.use_structure):
            raise ConfigError(
                "at least one modality must be enabled; a model with neither "
                "has no input"
            )
        if not 0 <= self.dropout < 1:
            raise ConfigError(f"dropout must be in [0, 1), got {self.dropout}")

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def build_model(config: TransformerConfig):
    """Construct the model. Imported lazily so torch stays optional."""
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as functional

    class DistanceBias(nn.Module):
        """Learned attention bias from pairwise CA-CA distances.

        Distances are expanded into radial basis functions before the linear
        map. A single scalar distance fed to a linear layer could only produce
        a monotonic bias — nearer always more attended — but the useful
        relationship is not monotonic: residues lining the same pocket sit at a
        characteristic separation, while both very close (sequence neighbours)
        and very distant residues are less informative. An RBF expansion lets
        the model learn a preferred distance band per head.
        """

        def __init__(self, n_heads: int, n_bases: int = 16, cutoff: float = 24.0):
            super().__init__()
            self.n_heads = n_heads
            self.cutoff = cutoff
            self.register_buffer("centres", torch.linspace(0.0, cutoff, n_bases))
            self.width = cutoff / max(n_bases - 1, 1)
            self.project = nn.Linear(n_bases, n_heads)
            # Start near zero so the bias does not dominate before training.
            nn.init.zeros_(self.project.bias)
            nn.init.normal_(self.project.weight, std=0.02)

        def forward(self, distances):
            # distances: (B, L, L) -> bias: (B, heads, L, L)
            clamped = distances.clamp(max=self.cutoff).unsqueeze(-1)
            rbf = torch.exp(-((clamped - self.centres) ** 2) / (2 * self.width ** 2))
            return self.project(rbf).permute(0, 3, 1, 2)

    class BiasedSelfAttention(nn.Module):
        """Multi-head self-attention that accepts an additive logit bias."""

        def __init__(self, d_model: int, n_heads: int, dropout: float):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.out = nn.Linear(d_model, d_model)
            self.dropout = dropout

        def forward(self, x, bias=None, key_padding_mask=None):
            batch, length, d_model = x.shape
            qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)

            mask = None
            if bias is not None:
                mask = bias
            if key_padding_mask is not None:
                # (B, 1, 1, L): padded keys are excluded from every query.
                pad = key_padding_mask[:, None, None, :]
                fill = torch.zeros_like(pad, dtype=x.dtype).masked_fill(
                    pad, float("-inf")
                )
                mask = fill if mask is None else mask + fill

            attended = functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
            attended = attended.transpose(1, 2).reshape(batch, length, d_model)
            return self.out(attended)

    class CrossAttention(nn.Module):
        """One stream attends to the other."""

        def __init__(self, d_model: int, n_heads: int, dropout: float):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )

        def forward(self, query, context, key_padding_mask=None):
            out, _ = self.attention(
                query, context, context,
                key_padding_mask=key_padding_mask, need_weights=False,
            )
            return out

    class Block(nn.Module):
        """Self-attention, cross-attention, feed-forward. Pre-norm."""

        def __init__(self, config: TransformerConfig, cross: bool):
            super().__init__()
            d_model = config.d_model
            self.norm1 = nn.LayerNorm(d_model)
            self.self_attention = BiasedSelfAttention(
                d_model, config.n_heads, config.dropout
            )
            self.cross = cross
            if cross:
                self.norm2 = nn.LayerNorm(d_model)
                self.cross_attention = CrossAttention(
                    d_model, config.n_heads, config.dropout
                )
            self.norm3 = nn.LayerNorm(d_model)
            hidden = d_model * config.ff_multiplier
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, hidden), nn.GELU(),
                nn.Dropout(config.dropout), nn.Linear(hidden, d_model),
            )
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x, context=None, bias=None, key_padding_mask=None):
            x = x + self.dropout(
                self.self_attention(self.norm1(x), bias, key_padding_mask)
            )
            if self.cross and context is not None:
                x = x + self.dropout(
                    self.cross_attention(
                        self.norm2(x), context, key_padding_mask
                    )
                )
            return x + self.dropout(self.feed_forward(self.norm3(x)))

    class BindingSiteTransformer(nn.Module):
        """Per-residue binding-site predictor."""

        def __init__(self, config: TransformerConfig):
            super().__init__()
            self.config = config
            d_model = config.d_model

            self.sequence_embed = (
                nn.Linear(config.sequence_dim, d_model) if config.use_sequence else None
            )
            self.structure_embed = (
                nn.Linear(config.structure_dim, d_model) if config.use_structure else None
            )
            # Learned positional embedding: sequence position matters (termini,
            # motifs) but the relationship is not the smooth one sinusoidal
            # encodings assume for this task.
            self.positions = nn.Embedding(config.max_length, d_model)

            both = config.use_sequence and config.use_structure
            self.sequence_blocks = nn.ModuleList(
                [Block(config, cross=both) for _ in range(config.n_layers)]
            ) if config.use_sequence else None
            self.structure_blocks = nn.ModuleList(
                [Block(config, cross=both) for _ in range(config.n_layers)]
            ) if config.use_structure else None

            self.distance_bias = (
                DistanceBias(config.n_heads)
                if config.use_distance_bias and config.use_structure else None
            )

            fused = d_model * (2 if both else 1)
            self.head = nn.Sequential(
                nn.LayerNorm(fused), nn.Linear(fused, d_model), nn.GELU(),
                nn.Dropout(config.dropout), nn.Linear(d_model, 1),
            )

        def forward(self, sequence_features, structure_features, distances=None, mask=None):
            """Returns per-residue logits of shape (batch, length).

            ``mask`` is True for real residues and False for padding.
            """
            reference = (
                sequence_features if sequence_features is not None else structure_features
            )
            batch, length = reference.shape[:2]
            if length > self.config.max_length:
                raise ValueError(
                    f"sequence length {length} exceeds max_length "
                    f"{self.config.max_length}; raise max_length or crop the chain"
                )
            key_padding = None if mask is None else ~mask

            index = torch.arange(length, device=reference.device)
            positional = self.positions(index)[None, :, :]

            bias = None
            if self.distance_bias is not None and distances is not None:
                bias = self.distance_bias(distances)

            sequence_stream = structure_stream = None
            if self.sequence_embed is not None:
                sequence_stream = self.sequence_embed(sequence_features) + positional
            if self.structure_embed is not None:
                structure_stream = self.structure_embed(structure_features) + positional

            for depth in range(self.config.n_layers):
                # Both streams read the *previous* layer's other stream, so
                # neither gets a within-layer advantage from ordering.
                previous_sequence = sequence_stream
                previous_structure = structure_stream
                if sequence_stream is not None:
                    sequence_stream = self.sequence_blocks[depth](
                        sequence_stream, previous_structure, bias, key_padding
                    )
                if structure_stream is not None:
                    structure_stream = self.structure_blocks[depth](
                        structure_stream, previous_sequence, bias, key_padding
                    )

            streams = [s for s in (sequence_stream, structure_stream) if s is not None]
            fused = torch.cat(streams, dim=-1) if len(streams) > 1 else streams[0]
            return self.head(fused).squeeze(-1)

        def n_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return BindingSiteTransformer(config)


def select_device(preference: str = "auto") -> str:
    """Choose a compute device.

    ``auto`` prefers CUDA, then Apple MPS, then CPU. MPS is included because
    it is the only accelerator available on an Apple laptop, and it makes the
    difference between a model that can be trained here and one that can only
    be described.
    """
    torch = _torch()
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
