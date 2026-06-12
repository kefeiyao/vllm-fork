# Fused Q/K RMSNorm Switch (`VLLM_USE_FUSED_QK_RMSNORM`)

This guide explains how to enable and validate the fused Q/K RMSNorm kernels in
the **HY V3** model (`vllm/model_executor/models/hy_v3.py`). It is intended for
testers who want to compare the default implementation against the fused kernels.

---

## 1. What the switch does

The behavior of `HYV3Attention` is selected by the environment variable
`VLLM_USE_FUSED_QK_RMSNORM`:

| Value         | Behavior                                                                                       | Kernel source |
| ------------- | ---------------------------------------------------------------------------------------------- | ------------- |
| `0` / unset   | **Default.** Separate per-head `q_norm` / `k_norm` (`RMSNorm`), then RoPE applied separately.  | pure PyTorch  |
| `1`           | **Fused Q/K RMSNorm only.** One kernel normalizes the Q and K regions of `qkv` in-place; RoPE is still applied afterwards. | standalone `qk_rms_norm_ops` SYCL/XPU extension |
| `2`           | **Fused Q/K RMSNorm + RoPE.** One kernel applies both RMSNorm and RoPE in-place on `qkv`; the separate RoPE call is skipped. | `torch.ops._C.fused_qk_norm_rope` (vllm-xpu-kernels) |

Any unrecognized value logs a warning and falls back to mode `0`.

**Fail-safe:** if the requested kernel is not available at model
initialization, the model logs a warning and silently falls back to mode `0`
(the default PyTorch path). It will **not** crash. Always check the logs to
confirm which path is actually active (see §5).

---

## 2. Quick start

```bash

# --- Mode 0: default (baseline) ---
unset VLLM_USE_FUSED_QK_RMSNORM

# --- Mode 1: fused Q/K RMSNorm only ---
export VLLM_USE_FUSED_QK_RMSNORM=1
export VLLM_QK_RMS_NORM_OPS_PATH=/abs/path/to/qk_rms_norm_release/release

# --- Mode 2: fused Q/K RMSNorm + RoPE ---
export VLLM_USE_FUSED_QK_RMSNORM=2
# (requires vllm-xpu-kernels installed in this environment)
```

Then launch your normal serving / benchmark command for the HY V3 model.

---


## 3. How to confirm which path is active

Grep the worker logs at startup:

| Log line                                                              | Meaning                          |
| --------------------------------------------------------------------- | -------------------------------- |
| `Loaded fused Q/K RMSNorm extension from <path>`                      | Mode 1 active.                   |
| `Using fused Q/K RMSNorm+RoPE op (torch.ops._C).`                     | Mode 2 active.                   |
| `... could not be imported. Falling back to ...`                      | Requested mode 1 but **fell back to mode 0**. |
| `... 'fused_qk_norm_rope' op ... is not registered. Falling back ...` | Requested mode 2 but **fell back to mode 0**. |
| (no fused log lines)                                                  | Mode 0 (default).                |

> If you set the env var but see a fallback warning, treat the run as **mode 0**.
> Fix the kernel availability (§2) before trusting any comparison.

---

## 4. Where the code lives

- Switch logic and helpers: `vllm/model_executor/models/hy_v3.py`
  - `_fused_qk_rmsnorm_mode()` — parses the env var into `0/1/2`.
  - `_get_qk_rms_norm_ops()` / `_resolve_qk_rms_norm_so_path()` — load the mode-1
    `.so` via import or `VLLM_QK_RMS_NORM_OPS_PATH`.
  - `_get_fused_qk_norm_rope_op()` — resolve the mode-2 op.
  - `HYV3Attention.__init__` resolves the active mode once and caches the kernel
    handles; `HYV3Attention.forward` dispatches on `self._fused_mode`.

---

## 5. Environment variables summary

| Variable                      | Required for | Purpose                                                         |
| ----------------------------- | ------------ | --------------------------------------------------------------- |
| `VLLM_USE_FUSED_QK_RMSNORM`   | all          | Selects the path: `0` (default), `1` (fused norm), `2` (fused norm+RoPE). |
| `VLLM_QK_RMS_NORM_OPS_PATH`   | mode 1       | Path to `qk_rms_norm_ops.so` or its containing directory (alternative to `PYTHONPATH`). |
