# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mockingbird architecture (partial smoke-port from Parameter Golf).

Source: parameter-golf-lab/records/track_10min_16mb/2026-05-01_Mockingbird_8xH100/train_gpt.py
(reference run: 11L x 512, mlp_mult=3.75, 1.062 BPB val on FineWeb 10B at 600s/16MB).

FAITHFUL TO REFERENCE:
- leaky_relu(neg_slope=0.5)^2 MLP (NOT gated SwiGLU): hidden = leaky_relu(up_proj(x), 0.5)^2; down_proj(hidden)
- Per-block attn_scale, mlp_scale (per-dim learnable scalar gains on residual updates)
- RMSNorm pre-norm
- Tied embeddings
- Global scalar QK gain via AttentionConfig.scaling_factor = qk_gain / sqrt(head_dim)
- 11L x 512 dim 8 heads mlp_mult=3.75 defaults
- ln_scale_factor = 1/sqrt(layer_idx+1) (when ln_scale=True)

NOT YET PORTED (TODO, in roughly the order to add them):
- Per-head learnable QK gain (reference: nn.Parameter((num_heads,), qk_gain_init=5.25)).
  Current port uses a single scalar via AttentionConfig.scaling_factor.
- resid_mix: per-dim learnable 2-vec that mixes current x with embedding x0 at each block.
  Requires plumbing x0 through the layer scan; not in Llama's signature today.
- U-Net encoder/decoder skip connections with learnable skip_weights + skip_gates.
  Reference uses skip from encoder layer i to decoder layer N-i-1.
- Looped middle block (loop_start=3, loop_end=5, num_loops=2, enable_looping_at=0.45).
  Implementation plan: split layers into pre/loop/post Stacked blocks, with a
  step-conditional lax.cond toggling the loop.
- Logit softcap on the LM head.
- yarn rope, rope_dims.
- Forward-1 token smear of the embedding lane.

The smoke target is: init, forward, backward all work on CPU with TinyStories.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional, Type, cast

import equinox as eqx
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import BlockFoldable, BlockSeq, ScanCheckpointPolicy, Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.layers import LayerNormConfigBase, RmsNormConfig
from levanter.layers.attention import Attention, AttentionBackend, AttentionConfig, AttentionMask
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.flop_utils import lm_flops_per_token


def _leaky_relu_squared(x: NamedArray, negative_slope: float = 0.5) -> NamedArray:
    """leaky_relu(x, neg_slope=0.5)^2 — the Mockingbird MLP activation.

    hax.nn.leaky_relu doesn't take a slope, so we inline it as where(x > 0, x, x * slope).
    """
    y = hax.where(x > 0, x, x * negative_slope)
    return y * y


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
        # Scalar QK gain folded into scaling_factor (default 1/sqrt(head_dim) * qk_gain).
        # TODO Phase 2: replace with per-head learnable Param applied to Q after RoPE.
        # qk_norm = RMSNorm on Q,K before RoPE (reference: F.rms_norm on each).
        scaling = self.qk_gain / math.sqrt(self.head_size)
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            use_bias=self.use_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=self.rope,
            scaling_factor=scaling,
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
    self_attn: Attention
    mlp: MockingbirdMlp
    attn_norm: hnn.RmsNorm
    mlp_norm: hnn.RmsNorm
    attn_scale: Optional[NamedArray]
    mlp_scale: Optional[NamedArray]

    @staticmethod
    def init(config: MockingbirdConfig, *, key) -> "MockingbirdBlock":
        k_attn, k_mlp = jrandom.split(key, 2)
        attn_config = config.attention_config()
        attn = Attention.init(attn_config, key=k_attn)
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
