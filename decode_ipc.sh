#!/bin/bash
# Decode instance for disaggregated prefill-decode with IPC (NIXL connector)

# IPC environment
export UCX_MEMTYPE_CACHE=0
export UCX_TLS=self,tcp,ze_ipc,ze_copy
export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/.nixl.mesonpy.libs/plugins:${LD_LIBRARY_PATH}

# Defaults
MODEL="/host/hf_models/Qwen3-30B-A3B/"
TP=4
DP=1
EP=1
ALL2ALL_BACKEND="allgather_reducescatter"
QUANT=""
DEVICE_MASK="4,5,6,7"
ATTN_BACKEND="FLASH_ATTN"
DTYPE="float16"
PORT=8200
SIDE_CHANNEL_PORT=5587
NUMA_CONTROL=""
HYBRID_KV=0
KILL=0

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  -m, --model MODEL         Model path (default: $MODEL)"
    echo "  -t, --tp N                Tensor parallelism size (default: $TP)"
    echo "      --dp N                Data parallelism size (default: $DP)"
    echo "  -e, --ep                  Enable expert parallelism (default: on)"
    echo "      --no-ep               Disable expert parallelism"
    echo "  -a, --all2all BACKEND     All2all backend (default: $ALL2ALL_BACKEND)"
    echo "  -q, --quant METHOD        Quantization method (e.g. fp8)"
    echo "  -d, --devices MASK        ZE_AFFINITY_MASK (default: $DEVICE_MASK)"
    echo "  -b, --attn-backend BACK   Attention backend (default: $ATTN_BACKEND)"
    echo "                            Valid: FLASH_ATTN, TRITON_ATTN, TORCH_SDPA"
    echo "      --dtype DTYPE         Data type (default: $DTYPE)"
    echo "  -p, --port PORT           Serve port (default: $PORT)"
    echo "      --hybrid-kv            Enable hybrid KV cache manager"
    echo "  -n, --numa SPEC           NUMA binding (cpulist:memnode per rank, ';'-separated)"
    echo "  -k, --kill                Kill running vllm processes before starting"
    echo "  -h, --help                Show this help"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)   MODEL="$2"; shift 2 ;;
        -t|--tp)      TP="$2"; shift 2 ;;
        --dp)         DP="$2"; shift 2 ;;
        -e|--ep)      EP=1; shift ;;
        --no-ep)      EP=0; shift ;;
        -a|--all2all) ALL2ALL_BACKEND="$2"; shift 2 ;;
        -q|--quant)   QUANT="$2"; shift 2 ;;
        -d|--devices) DEVICE_MASK="$2"; shift 2 ;;
        -b|--attn-backend) ATTN_BACKEND="$2"; shift 2 ;;
        --dtype)      DTYPE="$2"; shift 2 ;;
        -p|--port)    PORT="$2"; shift 2 ;;
        --hybrid-kv)  HYBRID_KV=1; shift ;;
        -n|--numa)    NUMA_CONTROL="$2"; shift 2 ;;
        -k|--kill)    KILL=1; shift ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Validity checks
if [[ ! "$TP" =~ ^[0-9]+$ ]] || [[ "$TP" -lt 1 ]]; then
    echo "Error: --tp must be a positive integer (got: $TP)"; exit 1
fi

case "$ATTN_BACKEND" in
    FLASH_ATTN|TRITON_ATTN|TORCH_SDPA) ;;
    *) echo "Error: --attn-backend must be one of: FLASH_ATTN, TRITON_ATTN, TORCH_SDPA (got: $ATTN_BACKEND)"; exit 1 ;;
esac

if [[ "$KILL" -eq 1 ]]; then
    echo "Killing existing vllm processes..."
    pkill -9 -f "vllm serve" 2>/dev/null
    pkill -9 -f "VLLM::" 2>/dev/null
    pkill -9 -f "python" 2>/dev/null
    sleep 2
    remaining=$(pgrep -f "VLLM::|vllm serve" 2>/dev/null | wc -l)
    if [[ "$remaining" -gt 0 ]]; then
        echo "Warning: $remaining vllm process(es) still running, force killing by PID..."
        pgrep -f "VLLM::|vllm serve" 2>/dev/null | xargs -r kill -9 2>/dev/null
    fi
    echo "Done."
    exit 0
fi

export ZE_AFFINITY_MASK="$DEVICE_MASK"

EP_FLAG=""
if [[ "$EP" -eq 1 ]]; then
    EP_FLAG="--enable-expert-parallel"
fi

QUANT_FLAG=""
if [[ -n "$QUANT" ]]; then
    QUANT_FLAG="--quantization $QUANT"
fi

HYBRID_KV_FLAG=""
if [[ "$HYBRID_KV" -eq 1 ]]; then
    HYBRID_KV_FLAG="--no-disable-hybrid-kv-cache-manager"
fi

if [[ -n "$NUMA_CONTROL" ]]; then
    export VLLM_NUMA_CONTROL="$NUMA_CONTROL"
    echo "NUMA binding: $NUMA_CONTROL"
fi

echo "Starting DECODE instance on port $PORT (devices: $DEVICE_MASK)"

VLLM_USE_V1=1 VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT="$SIDE_CHANNEL_PORT" VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        vllm serve "$MODEL" \
        --tensor-parallel-size "$TP" \
        --data-parallel-size "$DP" \
        $EP_FLAG \
        --all2all-backend "$ALL2ALL_BACKEND" \
        --host localhost \
        --port "$PORT" \
        --seed 42 \
        --enforce-eager \
        --dtype "$DTYPE" \
        --gpu-memory-utilization 0.9 \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"xpu"}' \
        $HYBRID_KV_FLAG \
        --max-model-len 8192 \
        --block-size 64 \
        --attention-backend "$ATTN_BACKEND" \
        --no-enable-prefix-caching \
        $QUANT_FLAG >decode.log 2>&1 &
