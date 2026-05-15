# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mockingbird architecture (partial port from Parameter Golf).

Source: parameter-golf-lab/records/track_10min_16mb/2026-05-01_Mockingbird_8xH100/train_gpt.py
(reference run: 11L x 512, mlp_mult=3.75, 1.062 BPB val on FineWeb 10B at 600s/16MB).

FAITHFUL TO REFERENCE (Phase 1a complete):
- leaky_relu(neg_slope=0.5)^2 MLP (NOT gated SwiGLU): hidden = leaky_relu(up_proj(x), 0.5)^2; down_proj(hidden)
- Per-block attn_scale, mlp_scale (per-dim learnable scalar gains on residual updates)
- RMSNorm pre-norm + QK rmsnorm pre-RoPE (via AttentionConfig.qk_norm)
- Tied embeddings, num_kv_heads=4 (GQA 8:4)
- Per-head learnable QK gain Param of shape (kv_head, q_heads_per_group),
  init 5.25, applied to q after RoPE (MockingbirdAttention)
- Partial RoPE: rotates first `rope_dims` of head dim only, rest pass-through,
  via zero-padded inv_freq (PartialRotaryEmbeddings)
- Logit softcap on LM head (cap*tanh(logits/cap))

NOT YET PORTED (TODO, in roughly the order to add them):
- ln_scale: norm output multiplied by 1/sqrt(layer_idx+1). Needs layer index
  plumbing through the layer Stacked.
- resid_mix: per-block Param (2,dim) mixes current x with embedding x0:
  x_in = mix[0]*x + mix[1]*x0. Needs x0 plumbing through fold kwargs.
- Smear gate: x_t += lam*sigmoid(W*x_t[:12]) * x_{t-1}, BOS-masked, after embed.
- U-Net encoder/decoder skip connections with learnable skip_weights + skip_gates.
- Looped middle block (loop_start=3, loop_end=5, num_loops=2, always-on for
  the port; reference toggles at 0.45 of training).
- Parallel lanes: decoder layers >= parallel_start_layer run a 2-lane block.
- Sparse attention gate (W_g (num_heads, gate_window=12), scale 0.5, on SDPA out).
- XSA: orthogonal value projection on attn output (`_xsa_efficient`), all layers.

Smoke target: init, forward, backward all work on GPU with TinyStories.
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional, Type, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import BlockFoldable, BlockSeq, ScanCheckpointPolicy, Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.layers import LayerNormConfigBase, RmsNormConfig
from levanter.layers.attention import (
    Attention,
    AttentionBackend,
    AttentionConfig,
    AttentionMask,
    dot_product_attention,
)
from levanter.layers.rotary import (
    DefaultRotaryEmbeddingsConfig,
    RotaryEmbeddings,
    RotaryEmbeddingsConfig,
)
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.flop_utils import lm_flops_per_token


def _leaky_relu_squared(x: NamedArray, negative_slope: float = 0.5) -> NamedArray:
    """leaky_relu(x, neg_slope=0.5)^2 — the Mockingbird MLP activation.

    hax.nn.leaky_relu doesn't take a slope, so we inline it as where(x > 0, x, x * slope).
    """
    y = hax.where(x > 0, x, x * negative_slope)
    return y * y


def _rotate_half(x: NamedArray, HeadSize: Axis) -> NamedArray:
    """Mirror of levanter.layers.rotary._rotate_half (private upstream).

    Rotates the second half of HeadSize to negate-and-prepend.
    """
    x1 = x[HeadSize, : HeadSize.size // 2]
    x2 = x[HeadSize, HeadSize.size // 2 :]
    return hax.concatenate(HeadSize, (-x2, x1))


@dataclass(frozen=True)
class PartialRotaryEmbeddingsConfig(RotaryEmbeddingsConfig):
    """Rotate only the first ``rope_dims`` of the head dim; pass the rest through.

    Implementation: standard RoPE inv_freq for the active band, then zero-pad
    the inactive band so cos(0)=1 and sin(0)=0 leave those dims unchanged
    after the q*cos + rotate_half(q)*sin combine.

    Reference: PG train_gpt.py uses rope_dims=16 on head_size=64 (rotate the
    first quarter only). This is a static-config knob — once chosen at init
    it does not change during training.
    """

    theta: float = 10000.0
    rope_dims: int = 16

    def build(self, HeadSize: Axis) -> RotaryEmbeddings:
        if self.rope_dims <= 0 or self.rope_dims > HeadSize.size:
            raise ValueError(
                f"rope_dims={self.rope_dims} must be in (0, {HeadSize.size}] for HeadSize={HeadSize.size}."
            )
        if self.rope_dims % 2 != 0:
            raise ValueError(f"rope_dims={self.rope_dims} must be even.")
        if (HeadSize.size - self.rope_dims) % 2 != 0:
            raise ValueError(
                f"HeadSize.size - rope_dims = {HeadSize.size - self.rope_dims} must be even."
            )
        return PartialRotaryEmbeddings(HeadSize, self.rope_dims, self)

    @classmethod
    def make_from_hf_config(cls, rope_theta: float, config: dict) -> "RotaryEmbeddingsConfig":
        return PartialRotaryEmbeddingsConfig(theta=rope_theta, rope_dims=config.get("rope_dims", 16))

    def to_hf_config(self) -> tuple[float, dict | None]:
        return self.theta, {"rope_type": "partial", "rope_dims": self.rope_dims}


# Register so draccus can dispatch by name; safe re-registration is a no-op.
try:
    RotaryEmbeddingsConfig.register_subclass("partial", PartialRotaryEmbeddingsConfig)
except Exception:
    pass


class PartialRotaryEmbeddings(RotaryEmbeddings):
    HeadDim: Axis = eqx.field(static=True)
    rope_dims: int = eqx.field(static=True)
    config: PartialRotaryEmbeddingsConfig = eqx.field(static=True)

    def __call__(self, q: NamedArray, position_ids: NamedArray) -> NamedArray:
        with jax.ensure_compile_time_eval():
            head_dim = self.HeadDim.size
            ActiveHalf = self.HeadDim.resize(self.rope_dims // 2)
            inv_freq_active: NamedArray = 1.0 / (
                self.config.theta ** (hax.arange(ActiveHalf, step=2) / self.rope_dims)
            )
            HalfHead = self.HeadDim.resize(head_dim // 2)
            if self.rope_dims < head_dim:
                PassiveHalf = self.HeadDim.resize((head_dim - self.rope_dims) // 2)
                zero_pad = hax.zeros(PassiveHalf)
                # Rename slices to share an axis so concatenate works on HalfHead.
                inv_freq_active_h = inv_freq_active.rename({ActiveHalf.name: HalfHead.name})
                zero_pad_h = zero_pad.rename({PassiveHalf.name: HalfHead.name})
                inv_freq = hax.concatenate(HalfHead, (inv_freq_active_h, zero_pad_h))
            else:
                inv_freq = inv_freq_active.rename({ActiveHalf.name: HalfHead.name})

        freqs = inv_freq.broadcast_axis(position_ids.axes) * position_ids
        emb = hax.concatenate(self.HeadDim, (freqs, freqs))
        cos = hax.cos(emb).astype(q.dtype)
        sin = hax.sin(emb).astype(q.dtype)
        return q * cos + _rotate_half(q, self.HeadDim) * sin


@LmConfig.register_subclass("mockingbird")
@dataclass(frozen=True)
class MockingbirdConfig(LmConfig):
    """Config for the Mockingbird model.

    Defaults match the 2026-05-01 reference run.
    """

    # Core sizing (reference run defaults: 11L x 512, 8 heads, 4 kv heads, mlp_mult=3.75)
    max_seq_len: int = 1024
    hidden_dim: int = 512
    num_layers: int = 11
    num_heads: int = 8
    num_kv_heads: int = 4  # reference run uses GQA 8:4 (was 8 in the smoke port; fixed)
    mlp_mult: float = 3.75  # hidden_dim of MLP = mlp_mult * hidden_dim

    # Mockingbird-specific
    qk_gain: float = 5.25  # per-head learnable Parameter init, applied to Q after RoPE
    leaky_relu_neg_slope: float = 0.5
    use_attn_scale: bool = True  # per-block per-dim learnable gain on attn residual
    use_mlp_scale: bool = True  # per-block per-dim learnable gain on mlp residual
    ln_scale: bool = True  # multiply norm output by 1/sqrt(layer_idx+1); reference: ON

    # Structural / Phase-2 features (parsed; wired by the custom transformer skeleton).
    loop_start: int = 3
    loop_end: int = 5  # inclusive
    num_loops: int = 2
    enable_looping: bool = True  # static config; reference toggles at 0.45 of training
    skip_gates_enabled: bool = True
    parallel_start_layer: int = 8  # decoder layers >= psl use the parallel-lane block
    smear_gate_enabled: bool = True
    smear_window: int = 12

    # LM-head logit softcap (cap*tanh(logits/cap)). Reference: 30.0. <=0 disables.
    logit_softcap: float = 30.0

    # Partial RoPE: rotate only the first `rope_dims` of each head; the rest unrotated.
    # Reference run uses 16; 0 means full RoPE over head_dim.
    rope_dims: int = 16

    # Standard
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = True
    use_bias: bool = False

    # Attention plumbing
    upcast_attn: bool = False
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None
    rope: RotaryEmbeddingsConfig = dataclasses.field(default_factory=DefaultRotaryEmbeddingsConfig)

    gradient_checkpointing: bool | ScanCheckpointPolicy | str = True
    scan_layers: bool = True

    tokenizer: Optional[str] = None

    def __post_init__(self):
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}."
        )
        if self.enable_looping:
            assert 0 <= self.loop_start <= self.loop_end < self.num_layers, (
                f"Invalid loop range [{self.loop_start}, {self.loop_end}] for num_layers={self.num_layers}."
            )

    @property
    def Pos(self) -> Axis:
        return Axis(name="position", size=self.max_seq_len)

    @property
    def KeyPos(self) -> Axis:
        return Axis(name="key_position", size=self.max_seq_len)

    @property
    def Embed(self) -> Axis:
        return Axis(name="embed", size=self.hidden_dim)

    @property
    def Layers(self) -> Axis:
        return Axis(name="layer", size=self.num_layers)

    @property
    def Mlp(self) -> Axis:
        return Axis(name="mlp", size=int(self.mlp_mult * self.hidden_dim))

    @property
    def head_size(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def norm_config(self) -> LayerNormConfigBase:
        return RmsNormConfig(
            use_weight=True,
            use_bias=self.use_bias,
            eps=self.layer_norm_epsilon,
        )

    def mk_LayerNorm(self, axis: AxisSpec):
        return self.norm_config.build(axis)

    def attention_config(self) -> AttentionConfig:
        # qk_norm = RMSNorm on Q,K pre-RoPE (Levanter applies it inside _compute_qkv
        # before the rope call — verified at attention.py:1828-1836). Per-head QK gain
        # lives on MockingbirdAttention as a Param applied to q post-RoPE; the
        # remaining 1/sqrt(head_size) scaling is left to dot_product_attention's
        # default (scaling_factor=None).
        rope_cfg: RotaryEmbeddingsConfig
        if 0 < self.rope_dims < self.head_size:
            rope_cfg = PartialRotaryEmbeddingsConfig(theta=10000.0, rope_dims=self.rope_dims)
        else:
            rope_cfg = self.rope
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            use_bias=self.use_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=rope_cfg,
            scaling_factor=None,
            qk_norm=self.norm_config,
        )

    def flops_per_token(self, vocab_size: int, context_length: int):
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=int(self.mlp_mult * self.hidden_dim),
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=vocab_size,
            glu=False,  # leaky-relu^2 MLP has only up_proj + down_proj, no gate_proj
        )

    @property
    def model_type(self) -> Type["MockingbirdLMHeadModel"]:
        return MockingbirdLMHeadModel


class MockingbirdMlp(eqx.Module):
    """leaky_relu(up_proj(x), neg_slope=0.5)^2 -> down_proj(hidden).

    Reference: train_gpt.py line 1186.
    Note: NOT gated. There is no gate_proj. Only up_proj and down_proj.
    """

    up_proj: hnn.Linear
    down_proj: hnn.Linear
    negative_slope: float = eqx.field(static=True)

    @staticmethod
    def init(
        Embed: AxisSpec, Mlp: AxisSpec, negative_slope: float, *, key, use_bias: bool = False
    ) -> "MockingbirdMlp":
        k_up, k_down = jrandom.split(key, 2)
        up_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_up, use_bias=use_bias, out_first=True)
        down_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_down, use_bias=use_bias, out_first=True)
        return MockingbirdMlp(up_proj, down_proj, negative_slope)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_up, k_down = maybe_rng_split(key, 2)
        h = self.up_proj(x, key=k_up)
        h = _leaky_relu_squared(h, negative_slope=self.negative_slope)
        return self.down_proj(h, key=k_down)


class MockingbirdAttention(Attention):
    """Levanter Attention + per-head learnable QK gain Param applied to q post-RoPE.

    The base class (configured with ``qk_norm``) handles QK rmsnorm pre-RoPE
    and RoPE; we add a learnable Param of shape (KVHeads, QHeadsPerGroup) that
    multiplies q after RoPE and before the SDPA call. This replaces the scalar
    ``AttentionConfig.scaling_factor`` and matches PG ``train_gpt.py`` per-head
    qk_gain (init 5.25).
    """

    # Defaults to None to satisfy dataclass field ordering (parent has defaulted
    # fields). init() always supplies the actual Param.
    per_head_qk_gain: Optional[NamedArray] = None

    @staticmethod
    def init(
        config: AttentionConfig, qk_gain_init: float, *, key
    ) -> "MockingbirdAttention":
        base = Attention.init(config, key=key)
        per_head_qk_gain = hax.full(
            (config.KVHeads, config.QHeadsPerGroup), qk_gain_init
        )
        return MockingbirdAttention(
            base.config,
            base.q_proj,
            base.k_proj,
            base.v_proj,
            base.o_proj,
            base.q_norm,
            base.k_norm,
            base.rot_embs,
            per_head_qk_gain,
        )

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: Optional[NamedArray | AttentionMask],
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        key_proj, key_o = maybe_rng_split(key, 2)
        q, k, v = self._compute_qkv(x, key=key_proj, pos_ids=pos_ids)

        q = q.rearrange((..., "kv_head", "q_heads_per_group", "position", "head_size"))
        k = k.rearrange((..., "kv_head", "position", "head_size"))
        v = v.rearrange((..., "kv_head", "position", "head_size"))
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        # Per-head QK gain applied to q (post-RoPE, pre-SDPA). The remaining
        # 1/sqrt(head_size) scaling is handled by dot_product_attention via
        # config.scaling_factor (None -> default 1/sqrt(head_size)).
        q = q * self.per_head_qk_gain

        if self.config.sliding_window is not None and isinstance(mask, AttentionMask):
            mask = mask.with_sliding_window(self.config.sliding_window)

        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask,
            attention_dtype=jnp.float32 if self.config.upcast_attn else x.dtype,
            attn_backend=self.config.attn_backend,
            flash_block_size=self.config.flash_attention_block_size,
            scaling_factor=self.config.scaling_factor,
            logits_soft_cap=self.config.logits_soft_cap,
            inference=True,
            prng=key,
        )
        attn_output = attn_output.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x.dtype)
        return self.o_proj(attn_output, key=key_o)


class MockingbirdBlock(eqx.Module):
    """Pre-norm transformer block with per-block per-dim attn/mlp residual scales.

    Reference block forward (train_gpt.py line 1232):
        attn_out = attn(attn_norm(x) * ln_scale_factor)
        x = x + attn_scale * attn_out
        mlp_out = mlp(mlp_norm(x) * ln_scale_factor)
        x = x + mlp_scale * mlp_out

    NOTE: resid_mix (mix of current x with embedding x0) is NOT yet plumbed; would require
    threading x0 through the scan-friendly layer signature. TODO once smoke is green.
    """

    config: MockingbirdConfig = eqx.field(static=True)
    self_attn: MockingbirdAttention
    mlp: MockingbirdMlp
    attn_norm: hnn.RmsNorm
    mlp_norm: hnn.RmsNorm
    attn_scale: Optional[NamedArray]
    mlp_scale: Optional[NamedArray]

    @staticmethod
    def init(config: MockingbirdConfig, *, key) -> "MockingbirdBlock":
        k_attn, k_mlp = jrandom.split(key, 2)
        attn_config = config.attention_config()
        attn = MockingbirdAttention.init(attn_config, qk_gain_init=config.qk_gain, key=k_attn)
        mlp = MockingbirdMlp.init(
            config.Embed,
            config.Mlp,
            negative_slope=config.leaky_relu_neg_slope,
            key=k_mlp,
            use_bias=config.use_bias,
        )
        attn_norm = config.mk_LayerNorm(config.Embed)
        mlp_norm = config.mk_LayerNorm(config.Embed)
        # Per-dim learnable scalar gains, initialized to ones.
        attn_scale = hax.ones(config.Embed) if config.use_attn_scale else None
        mlp_scale = hax.ones(config.Embed) if config.use_mlp_scale else None
        return MockingbirdBlock(config, attn, mlp, attn_norm, mlp_norm, attn_scale, mlp_scale)

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: Optional[NamedArray | AttentionMask],
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)

        # Attention path
        residual = x
        a = self.attn_norm(x)
        # NOTE: ln_scale_factor (1/sqrt(layer_idx+1)) NOT yet wired — needs per-layer idx in scan.
        attn_out = self.self_attn(x=a, mask=mask, key=k_attn, pos_ids=pos_ids)
        if self.attn_scale is not None:
            attn_out = attn_out * self.attn_scale
        x = residual + attn_out

        # MLP path
        residual = x
        m = self.mlp_norm(x)
        mlp_out = self.mlp(m, key=k_mlp)
        if self.mlp_scale is not None:
            mlp_out = mlp_out * self.mlp_scale
        x = residual + mlp_out
        return x


class MockingbirdTransformer(eqx.Module):
    config: MockingbirdConfig = eqx.field(static=True)
    layers: BlockFoldable[MockingbirdBlock]
    final_norm: hnn.RmsNorm

    @staticmethod
    def init(config: MockingbirdConfig, *, key) -> "MockingbirdTransformer":
        S = Stacked if config.scan_layers else BlockSeq
        layers = S.init(config.Layers, MockingbirdBlock, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        final_norm = config.mk_LayerNorm(config.Embed)
        return MockingbirdTransformer(config, layers, final_norm)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: Optional[NamedArray | AttentionMask], *, key, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        # TODO(loop): when enable_looping=True and current step >= enable_looping_at fraction,
        # split into pre/loop/post and run loop block num_loops+1 times. For now: single pass.
        x = cast(NamedArray, self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids))
        x = self.final_norm(x)
        return x


class MockingbirdEmbedding(ModuleWithStateDictSerialization, eqx.Module):
    token_embeddings: hnn.Embedding

    @staticmethod
    def init(Vocab: Axis, config: MockingbirdConfig, *, key) -> "MockingbirdEmbedding":
        token_embeddings = hnn.Embedding.init(Vocab, config.Embed, key=key)
        return MockingbirdEmbedding(token_embeddings)

    @property
    def Vocab(self) -> Axis:
        return self.token_embeddings.Vocab

    @property
    def Embed(self) -> Axis:
        return cast(Axis, self.token_embeddings.Embed)

    @named_call
    def embed(self, input_ids, *args):
        return self.token_embeddings(input_ids)

    def unembed(self, x: NamedArray):
        return self.token_embeddings.unembed(x)

    def resize_embeddings(self, new_size: int, key=None):
        new_weights = self.token_embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, token_embeddings=new_weights)


class MockingbirdLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[MockingbirdConfig]):
    transformer: MockingbirdTransformer
    embeddings: MockingbirdEmbedding
    lm_head: Optional[hnn.Linear]

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: MockingbirdConfig, *, key) -> "MockingbirdLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = MockingbirdTransformer.init(config, key=k_t)
        embeddings = MockingbirdEmbedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)
        return MockingbirdLMHeadModel(transformer, embeddings, lm_head)

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[NamedArray | AttentionMask] = None,
        pos_ids: NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        k_t, k_head = maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=k_t, pos_ids=pos_ids)
        if self.lm_head is not None:
            logits = self.lm_head(x, key=k_head)
        else:
            logits = self.embeddings.unembed(x)
        cap = self.config.logit_softcap
        if cap > 0.0:
            logits = cap * hax.tanh(logits / cap)
        return logits

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        x = self.embeddings.embed(input_ids)
        return self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        return self.lm_head.weight

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[MockingbirdConfig]":
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        if self.lm_head is not None:
            new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
            new_lm_head = dataclasses.replace(self.lm_head, Out=self.Vocab.resize(new_size), weight=new_lm_matrix)
            return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)
        return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self):
        return {"transformer": "model", "embeddings": None}
