# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3 model compatible with HuggingFace weights."""

import importlib.util
import os
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.encoder_only_attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsEagle, SupportsEagle3, SupportsLoRA, SupportsPP
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader, PPMissingLayer, extract_layer_index, maybe_prefix

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Optional fused Q/K RMSNorm kernels (XPU), selected via the environment
# variable ``VLLM_USE_FUSED_QK_RMSNORM``:
#   0 (default) : disabled, use the regular separate Q/K RMSNorm + RoPE.
#   1           : fused Q/K RMSNorm only, via the standalone ``qk_rms_norm_ops``
#                 SYCL/XPU extension (RoPE still applied separately afterwards).
#   2           : fused Q/K RMSNorm *and* RoPE, via the ``fused_qk_norm_rope``
#                 op registered by the vllm-xpu-kernels extension
#                 (``torch.ops._C.fused_qk_norm_rope``).
#
# Qwen3 applies QK-Norm *before* RoPE (norm-then-rope), which matches both
# fused kernels, so all three modes are mathematically equivalent here.
# ---------------------------------------------------------------------------
_qk_rms_norm_ops = None
_qk_rms_norm_import_failed = False


def _fused_qk_rmsnorm_mode() -> int:
    """Parse ``VLLM_USE_FUSED_QK_RMSNORM`` into an integer mode (0/1/2)."""
    raw = os.getenv("VLLM_USE_FUSED_QK_RMSNORM", "0").strip().lower()
    if raw == "0":
        return 0
    if raw == "1":
        return 1
    if raw == "2":
        return 2
    logger.warning(
        "Unrecognized VLLM_USE_FUSED_QK_RMSNORM=%r; expected 0, 1, or 2. "
        "Defaulting to 0 (disabled).",
        raw,
    )
    return 0


def _resolve_qk_rms_norm_so_path() -> str | None:
    """Resolve the path to ``qk_rms_norm_ops.so`` from the env var."""
    path = os.getenv("VLLM_QK_RMS_NORM_OPS_PATH")
    if not path:
        return None
    if os.path.isdir(path):
        candidate = os.path.join(path, "qk_rms_norm_ops.so")
        return candidate if os.path.isfile(candidate) else None
    return path if os.path.isfile(path) else None


def _get_qk_rms_norm_ops():
    """Lazily import the ``qk_rms_norm_ops`` extension (or return None).

    The extension is shipped as a standalone ``qk_rms_norm_ops.so`` rather than
    an installed package. We first try a normal import (works if its directory
    is on ``PYTHONPATH``), then fall back to loading the shared object directly
    from a path/directory given by ``VLLM_QK_RMS_NORM_OPS_PATH``.
    """
    global _qk_rms_norm_ops, _qk_rms_norm_import_failed
    if _qk_rms_norm_ops is not None or _qk_rms_norm_import_failed:
        return _qk_rms_norm_ops

    # 1) Normal import (directory on PYTHONPATH / installed package).
    try:
        import qk_rms_norm_ops  # type: ignore

        _qk_rms_norm_ops = qk_rms_norm_ops
        return _qk_rms_norm_ops
    except ImportError:
        pass

    # 2) Load the shared object directly from a user-provided location.
    so_path = _resolve_qk_rms_norm_so_path()
    if so_path is not None:
        try:
            spec = importlib.util.spec_from_file_location(
                "qk_rms_norm_ops", so_path
            )
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _qk_rms_norm_ops = module
                logger.info("Loaded fused Q/K RMSNorm extension from %s", so_path)
                return _qk_rms_norm_ops
        except Exception as e:  # noqa: BLE001 - report and fall back
            logger.warning(
                "Failed to load fused Q/K RMSNorm extension from %s: %s",
                so_path,
                e,
            )

    _qk_rms_norm_import_failed = True
    logger.warning(
        "VLLM_USE_FUSED_QK_RMSNORM is set but the 'qk_rms_norm_ops' extension "
        "could not be imported. Set VLLM_QK_RMS_NORM_OPS_PATH to the "
        "'qk_rms_norm_ops.so' file (or its directory), or add that directory "
        "to PYTHONPATH. Falling back to the default separate Q/K RMSNorm "
        "implementation."
    )
    return _qk_rms_norm_ops


_fused_qk_norm_rope_op = None
_fused_qk_norm_rope_lookup_failed = False


def _get_fused_qk_norm_rope_op():
    """Return ``torch.ops._C.fused_qk_norm_rope`` if available, else None.

    This op is registered by the vllm-xpu-kernels extension and fuses Q/K
    RMSNorm together with RoPE in a single kernel.
    """
    global _fused_qk_norm_rope_op, _fused_qk_norm_rope_lookup_failed
    if _fused_qk_norm_rope_op is not None or _fused_qk_norm_rope_lookup_failed:
        return _fused_qk_norm_rope_op

    # Best-effort import to ensure custom ops are registered under torch.ops._C.
    try:
        import vllm._C  # type: ignore  # noqa: F401
    except ImportError:
        pass

    try:
        op = torch.ops._C.fused_qk_norm_rope
        # Touch the op to make sure it is actually registered.
        _ = op.default
        _fused_qk_norm_rope_op = op
        logger.info("Using fused Q/K RMSNorm+RoPE op (torch.ops._C).")
        return _fused_qk_norm_rope_op
    except (AttributeError, RuntimeError):
        _fused_qk_norm_rope_lookup_failed = True
        logger.warning(
            "VLLM_USE_FUSED_QK_RMSNORM=2 requested but the "
            "'fused_qk_norm_rope' op (vllm-xpu-kernels) is not registered. "
            "Falling back to the default separate Q/K RMSNorm + RoPE path."
        )
        return None


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.rms_norm_eps = rms_norm_eps

        # Resolve the optional fused-kernel mode (0/1/2) once, at init.
        #   1 -> fused Q/K RMSNorm only (qk_rms_norm_ops)
        #   2 -> fused Q/K RMSNorm + RoPE (torch.ops._C.fused_qk_norm_rope)
        # If the requested kernel is unavailable we fall back to mode 0.
        self._fused_mode = 0
        self._qk_rms_norm_ops = None
        self._fused_qk_norm_rope_op = None
        requested_mode = _fused_qk_rmsnorm_mode()
        if requested_mode == 1:
            ops = _get_qk_rms_norm_ops()
            if ops is not None:
                self._qk_rms_norm_ops = ops
                self._fused_mode = 1
        elif requested_mode == 2:
            op = _get_fused_qk_norm_rope_op()
            if op is not None:
                self._fused_qk_norm_rope_op = op
                self._fused_mode = 2
        # Float32 copies of the norm weights required by the mode-1 kernel;
        # built lazily on first forward (after weights are loaded). The
        # mode-2 kernel consumes the weights in the model dtype directly.
        self._q_norm_weight_f32: torch.Tensor | None = None
        self._k_norm_weight_f32: torch.Tensor | None = None

    def _maybe_init_fused_qk_weights(self) -> None:
        """Build float32 copies of the q/k norm weights for the kernel.

        If the weights are already float32 we skip the dtype conversion and
        only ensure contiguity.
        """
        w = self.q_norm.weight
        if (
            self._q_norm_weight_f32 is None
            or self._q_norm_weight_f32.device != w.device
        ):
            qw = self.q_norm.weight.detach()
            if qw.dtype != torch.float32:
                qw = qw.to(torch.float32)
            self._q_norm_weight_f32 = qw.contiguous()
            kw = self.k_norm.weight.detach()
            if kw.dtype != torch.float32:
                kw = kw.to(torch.float32)
            self._k_norm_weight_f32 = kw.contiguous()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        rope_applied = False
        if self._fused_mode == 1:
            # Mode 1: fused kernel normalizes the Q and K head regions of `qkv`
            # in-place (V untouched). RoPE is still applied separately below.
            qkv = qkv.contiguous()
            self._maybe_init_fused_qk_weights()
            self._qk_rms_norm_ops.qk_rms_norm_forward(
                qkv,
                self._q_norm_weight_f32,
                self._k_norm_weight_f32,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.head_dim,
                self.rms_norm_eps,
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        elif self._fused_mode == 2:
            # Mode 2: fused kernel applies Q/K RMSNorm *and* RoPE in-place on
            # `qkv` (V untouched), so we skip the separate rotary_emb call.
            qkv = qkv.contiguous()
            position_ids = positions
            if position_ids.dtype != torch.int64:
                position_ids = position_ids.to(torch.int64)
            position_ids = position_ids.contiguous()
            self._fused_qk_norm_rope_op(
                qkv,
                self.num_heads,
                self.num_kv_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rms_norm_eps,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                self.rotary_emb.is_neox_style,
                position_ids,
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            rope_applied = True
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            # Add qk-norm
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)
        if not rope_applied:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )


class Qwen3ForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
