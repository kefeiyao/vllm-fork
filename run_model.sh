#!/bin/bash
#export VLLM_LOGGING_LEVEL=debug

# Defaults
MODEL="/host/hf_models/Qwen3-30B-A3B/"
TP=4
DP=1
EP=1
ALL2ALL_BACKEND="allgather_reducescatter"
QUANT=""
DEVICE_MASK="3,4,5,6"
ATTN_BACKEND="FLASH_ATTN"
DTYPE="bfloat16"
PORT=8100
NUMA_CONTROL=""
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
    echo "  -n, --numa SPEC           NUMA binding (cpulist:memnode per rank, ';'-separated)"
    echo "                            Example: '0-15,64-79:0;16-31,80-95:0;32-47,96-111:1;48-63,112-127:1'"
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

if [[ -n "$QUANT" ]]; then
    case "$QUANT" in
        fp8|awq|gptq|squeezellm|marlin|fp8_e5m2|fp8_e4m3) ;;
        *) echo "Warning: unrecognized quantization method '$QUANT', proceeding anyway" ;;
    esac
fi

case "$ALL2ALL_BACKEND" in
    allgather_reducescatter|pplx|deepep_low_latency|deepep_high_throughput) ;;
    *) echo "Warning: unrecognized all2all backend '$ALL2ALL_BACKEND', proceeding anyway" ;;
esac

if [[ "$KILL" -eq 1 ]]; then
    echo "Killing existing vllm processes..."
    # Force kill any remaining
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
    EP_FLAG="-ep"
fi

QUANT_FLAG=""
if [[ -n "$QUANT" ]]; then
    QUANT_FLAG="--quantization $QUANT"
fi

NUMA_ENV=""
if [[ -n "$NUMA_CONTROL" ]]; then
    export VLLM_NUMA_CONTROL="$NUMA_CONTROL"
    echo "NUMA binding: $NUMA_CONTROL"
fi

DP_FLAG=""
if [[ "$DP" -gt 1 ]]; then
    DP_FLAG="--data-parallel-size $DP"
fi

echo "Launching: MODEL=$MODEL TP=$TP DP=$DP EP=$EP DEVICES=$DEVICE_MASK PORT=$PORT DTYPE=$DTYPE ATTN=$ATTN_BACKEND ALL2ALL=$ALL2ALL_BACKEND"

VLLM_USE_V1=1 VLLM_NIXL_SIDE_CHANNEL_HOST=localhost VLLM_NIXL_SIDE_CHANNEL_PORT=5577 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        vllm serve "$MODEL" \
        -tp "$TP" \
        $DP_FLAG \
        $EP_FLAG \
        --all2all-backend "$ALL2ALL_BACKEND" \
        --host localhost \
        --port "$PORT" \
        --seed 42 \
        --enforce-eager \
        --dtype "$DTYPE" \
        --gpu-memory-utilization 0.8 \
        --max-model-len 8192 \
        --block-size 64 \
        --attention-backend "$ATTN_BACKEND" \
        --no-enable-prefix-caching \
        $QUANT_FLAG >run_model.log 2>&1 &

