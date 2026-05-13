#!/bin/bash
# Unified vLLM serve launcher
#
# Modes:
#   --holistic            Single (non-disaggregated) vLLM server
#   --prefill / --decode  Disaggregated prefill-decode instances (NIXL via UCX IPC or verbs)
#   --proxy               Disagg proxy server
#   --kill                Kill all running vllm/proxy processes

set -euo pipefail

# ── Log directory & rotation ──
LOG_DIR="vllm_logs/server"
LOG_MAX_BYTES=$((10 * 1024 * 1024))  # 10 MB

mkdir -p "$LOG_DIR"

rotate_log() {
    local logfile="$1"
    if [[ -f "$logfile" ]]; then
        local size
        size=$(stat -c%s "$logfile" 2>/dev/null || echo 0)
        if [[ "$size" -ge "$LOG_MAX_BYTES" ]]; then
            local ts
            ts=$(date +%Y%m%d_%H%M%S)
            mv "$logfile" "${logfile%.log}.${ts}.log"
            echo "  Rotated $logfile -> ${logfile%.log}.${ts}.log"
        fi
    fi
}

# ── Mode ──
MODE=""          # holistic | prefill | decode | proxy | kill

# ── Defaults ──
MODEL="/host/hf_models/Qwen3-30B-A3B/"
TP=4
DP=1
EP=1
ALL2ALL_BACKEND="allgather_reducescatter"
QUANT=""
DEVICE_MASK=""   # auto-set per mode if not given
ATTN_BACKEND="FLASH_ATTN"
DTYPE="float16"
PORT=""          # auto-set per mode if not given
NUMA_CONTROL=""
HYBRID_KV=0
PROFILE=0
PREFIX_CACHING=1
PROXY_WAIT=0
MAX_MODEL_LEN=8192
BLOCK_SIZE=64
GPU_MEM_UTIL=0.8
PROXY_PORT=8300
NIXL_TRANSPORT="ipc"
UCX_TLS_OVERRIDE=""
UCX_NET_DEVICES=""
SERVE_HOST="localhost"
SIDE_CHANNEL_HOST="localhost"
PROXY_HOST="localhost"
PREFILL_BACKEND_HOST="localhost"
PREFILL_BACKEND_PORT=8100
DECODE_BACKEND_HOST="localhost"
DECODE_BACKEND_PORT=8200

# Side-channel ports (internal, per disagg mode)
P_SIDE_CHANNEL_PORT=5577
D_SIDE_CHANNEL_PORT=5587

usage() {
    cat <<EOF
Usage: $0 MODE [OPTIONS]

Modes (exactly one required):
  --holistic            Start a single (non-disaggregated) vLLM server
  --prefill             Start prefill instance (disaggregated)
  --decode              Start decode instance (disaggregated)
  --proxy               Start the disagg proxy server
  --kill                Kill all running vllm/proxy processes and exit

Options:
  -m, --model MODEL       Model path (default: $MODEL)
  -t, --tp N              Tensor parallelism size (default: $TP)
      --dp N              Data parallelism size (default: $DP)
  -e, --ep                Enable expert parallelism (default: on)
      --no-ep             Disable expert parallelism
  -a, --all2all BACKEND   All2all backend (default: $ALL2ALL_BACKEND)
  -q, --quant METHOD      Quantization method (e.g. fp8)
  -d, --devices MASK      ZE_AFFINITY_MASK (default: per mode)
  -b, --attn-backend BACK Attention backend (default: $ATTN_BACKEND)
      --dtype DTYPE        Data type (default: $DTYPE)
  -p, --port PORT         Serve port (default: per mode)
      --hybrid-kv          Enable hybrid KV cache manager (disagg only)
      --profile            Enable torch profiler
      --no-prefix-caching  Disable prefix caching (default: on)
      --wait               Wait for prefill & decode to be ready (proxy only)
      --max-model-len N    Max model length (default: $MAX_MODEL_LEN)
      --block-size N       Block size (default: $BLOCK_SIZE)
      --gpu-mem-util F     GPU memory utilization (default: $GPU_MEM_UTIL)
  -n, --numa SPEC         NUMA binding (cpulist:memnode per rank, ';'-separated)
      --serve-host HOST    Host/IP for vLLM API bind (default: $SERVE_HOST)
      --nixl-transport MODE  Disagg NIXL transport preset: ipc|verbs (default: $NIXL_TRANSPORT)
      --ucx-tls TLS         Advanced: override the resolved UCX_TLS for disagg mode
      --ucx-net-devices DEV UCX_NET_DEVICES for disagg mode (default: auto; verbs=all)
      --side-channel-host H Host/IP advertised for NIXL side-channel (default: $SIDE_CHANNEL_HOST)
      --proxy-host HOST    Host/IP for proxy bind (default: $PROXY_HOST, with --proxy)
      --prefill-backend-host H  Proxy upstream prefiller host (default: $PREFILL_BACKEND_HOST)
      --prefill-backend-port P  Proxy upstream prefiller port (default: $PREFILL_BACKEND_PORT)
      --decode-backend-host H   Proxy upstream decoder host (default: $DECODE_BACKEND_HOST)
      --decode-backend-port P   Proxy upstream decoder port (default: $DECODE_BACKEND_PORT)
      --proxy-port PORT    Proxy listen port (default: $PROXY_PORT, with --proxy)
  -h, --help              Show this help

Mode defaults:
  --holistic   devices=0,1,2,3   port=8000
  --prefill    devices=0,1,2,3   port=8100
  --decode     devices=4,5,6,7   port=8200

Examples:
  # Holistic: single server, TP=4
  $0 --holistic -m \$model -t 4 -e -d 0,1,2,3 -q fp8 --dtype bfloat16

  # Prefill: TP=4 on GPUs 0-3
  $0 --prefill -m \$model -t 4 -e -b FLASH_ATTN -d 0,1,2,3 -q fp8 --dtype bfloat16

    # Prefill over UCX verbs on a multi-host setup
    $0 --prefill --serve-host 0.0.0.0 --side-channel-host <prefill_ip> \
         --nixl-transport verbs --ucx-net-devices all -m \$model -t 4 -d 0,1,2,3

  # Decode: TP=1, DP=4, veloci_deepep on GPUs 4-7
  $0 --decode -m \$model -t 1 --dp 4 -e -d 4,5,6,7 -q fp8 -a veloci_deepep --dtype bfloat16

    # Proxy for remote prefill/decode backends
    $0 --proxy --proxy-host 0.0.0.0 --prefill-backend-host <prefill_ip> --decode-backend-host <decode_ip>

  # Proxy (with wait for backends)
  $0 --proxy --wait

  # Kill everything
  $0 --kill
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        # ── Mode ──
        --holistic)        MODE="holistic"; shift ;;
        --prefill)         MODE="prefill"; shift ;;
        --decode)          MODE="decode"; shift ;;
        --proxy)           MODE="proxy"; shift ;;
        --kill|-k)         MODE="kill"; shift ;;

        # ── Options ──
        -m|--model)        MODEL="$2"; shift 2 ;;
        -t|--tp)           TP="$2"; TP_SET=1; shift 2 ;;
        --dp)              DP="$2"; shift 2 ;;
        -e|--ep)           EP=1; shift ;;
        --no-ep)           EP=0; shift ;;
        -a|--all2all)      ALL2ALL_BACKEND="$2"; shift 2 ;;
        -q|--quant)        QUANT="$2"; shift 2 ;;
        -d|--devices)      DEVICE_MASK="$2"; shift 2 ;;
        -b|--attn-backend) ATTN_BACKEND="$2"; shift 2 ;;
        --dtype)           DTYPE="$2"; shift 2 ;;
        -p|--port)         PORT="$2"; shift 2 ;;
        --hybrid-kv)       HYBRID_KV=1; shift ;;
        --profile)         PROFILE=1; shift ;;
        --no-prefix-caching) PREFIX_CACHING=0; shift ;;
        --wait)            PROXY_WAIT=1; shift ;;
        --max-model-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
        --block-size)      BLOCK_SIZE="$2"; shift 2 ;;
        --gpu-mem-util)    GPU_MEM_UTIL="$2"; shift 2 ;;
        -n|--numa)         NUMA_CONTROL="$2"; shift 2 ;;
        --serve-host)      SERVE_HOST="$2"; shift 2 ;;
        --nixl-transport)  NIXL_TRANSPORT="$2"; shift 2 ;;
        --ucx-tls)         UCX_TLS_OVERRIDE="$2"; shift 2 ;;
        --ucx-net-devices) UCX_NET_DEVICES="$2"; shift 2 ;;
        --side-channel-host) SIDE_CHANNEL_HOST="$2"; shift 2 ;;
        --proxy-host)      PROXY_HOST="$2"; shift 2 ;;
        --prefill-backend-host) PREFILL_BACKEND_HOST="$2"; shift 2 ;;
        --prefill-backend-port) PREFILL_BACKEND_PORT="$2"; shift 2 ;;
        --decode-backend-host) DECODE_BACKEND_HOST="$2"; shift 2 ;;
        --decode-backend-port) DECODE_BACKEND_PORT="$2"; shift 2 ;;
        --proxy-port)      PROXY_PORT="$2"; shift 2 ;;

        -h|--help)         usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── Kill mode ──
if [[ "$MODE" == "kill" ]]; then
    echo "Killing existing vllm/proxy processes..."
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "VLLM::" 2>/dev/null || true
    pkill -9 -f "toy_proxy_server" 2>/dev/null || true
    sleep 2
    remaining=$(pgrep -f "VLLM::|vllm serve|toy_proxy_server" 2>/dev/null | wc -l)
    if [[ "$remaining" -gt 0 ]]; then
        echo "Warning: $remaining process(es) still running, force killing..."
        pgrep -f "VLLM::|vllm serve|toy_proxy_server" 2>/dev/null | xargs -r kill -9 2>/dev/null
    fi
    echo "Done."
    exit 0
fi

# ── Proxy mode ──
if [[ "$MODE" == "proxy" ]]; then
    P_URL="http://${PREFILL_BACKEND_HOST}:${PREFILL_BACKEND_PORT}/health"
    D_URL="http://${DECODE_BACKEND_HOST}:${DECODE_BACKEND_PORT}/health"

    if [[ "$PROXY_WAIT" -eq 1 ]]; then
        echo "Waiting for prefill ($P_URL) and decode ($D_URL) to become ready..."
        P_READY=0
        D_READY=0
        while [[ "$P_READY" -eq 0 ]] || [[ "$D_READY" -eq 0 ]]; do
            if [[ "$P_READY" -eq 0 ]]; then
                if curl -sf "$P_URL" >/dev/null 2>&1; then
                    echo "  Prefill is ready."
                    P_READY=1
                fi
            fi
            if [[ "$D_READY" -eq 0 ]]; then
                if curl -sf "$D_URL" >/dev/null 2>&1; then
                    echo "  Decode is ready."
                    D_READY=1
                fi
            fi
            if [[ "$P_READY" -eq 0 ]] || [[ "$D_READY" -eq 0 ]]; then
                sleep 5
            fi
        done
        echo "Both servers ready."
    fi

    PROXY_LOG="$LOG_DIR/proxy.log"
    rotate_log "$PROXY_LOG"
    echo "Starting PROXY on ${PROXY_HOST}:$PROXY_PORT (prefill=${PREFILL_BACKEND_HOST}:${PREFILL_BACKEND_PORT}, decode=${DECODE_BACKEND_HOST}:${DECODE_BACKEND_PORT})"
    python3 /host/mnt/ctrl/disk1/kf/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
        --prefiller-host "$PREFILL_BACKEND_HOST" --prefiller-port "$PREFILL_BACKEND_PORT" \
        --decoder-host "$DECODE_BACKEND_HOST" --decoder-port "$DECODE_BACKEND_PORT" \
        --host "$PROXY_HOST" --port "$PROXY_PORT" \
        >>"$PROXY_LOG" 2>&1 &
    echo "  PID=$!, log=$PROXY_LOG"
    exit 0
fi

# ── Must be holistic, prefill, or decode ──
if [[ "$MODE" != "holistic" && "$MODE" != "prefill" && "$MODE" != "decode" ]]; then
    echo "Error: specify one of --holistic, --prefill, --decode, --proxy, --kill"
    usage
fi

# ── Apply mode-specific defaults ──
DISAGG=0
case "$MODE" in
    holistic)
        [[ -z "$DEVICE_MASK" ]] && DEVICE_MASK="0,1,2,3"
        [[ -z "$PORT" ]]        && PORT=8000
        LOG_FILE="$LOG_DIR/holistic.log"
        LABEL="HOLISTIC"
        ;;
    prefill)
        DISAGG=1
        [[ -z "$DEVICE_MASK" ]] && DEVICE_MASK="0,1,2,3"
        [[ -z "$PORT" ]]        && PORT=8100
        SIDE_CHANNEL_PORT=$P_SIDE_CHANNEL_PORT
        LOG_FILE="$LOG_DIR/prefill.log"
        LABEL="PREFILL"
        ;;
    decode)
        DISAGG=1
        [[ -z "$DEVICE_MASK" ]] && DEVICE_MASK="4,5,6,7"
        [[ -z "$PORT" ]]        && PORT=8200
        SIDE_CHANNEL_PORT=$D_SIDE_CHANNEL_PORT
        LOG_FILE="$LOG_DIR/decode.log"
        LABEL="DECODE"
        ;;
esac

# ── Default TP=1 when DP is enabled and --tp not explicitly set ──
if [[ "$DP" -gt 1 ]] && [[ "${TP_SET:-0}" -ne 1 ]]; then
    TP=1
fi

# ── Validation ──
if [[ ! "$TP" =~ ^[0-9]+$ ]] || [[ "$TP" -lt 1 ]]; then
    echo "Error: --tp must be a positive integer (got: $TP)"; exit 1
fi

case "$ATTN_BACKEND" in
    FLASH_ATTN|TRITON_ATTN|TORCH_SDPA) ;;
    *) echo "Error: --attn-backend must be one of: FLASH_ATTN, TRITON_ATTN, TORCH_SDPA (got: $ATTN_BACKEND)"; exit 1 ;;
esac

case "$NIXL_TRANSPORT" in
    ipc|verbs) ;;
    *) echo "Error: --nixl-transport must be one of: ipc, verbs (got: $NIXL_TRANSPORT)"; exit 1 ;;
esac

if [[ "$NIXL_TRANSPORT" == "verbs" ]] && [[ "$DISAGG" -eq 1 ]] && [[ "$SIDE_CHANNEL_HOST" == "localhost" ]]; then
    echo "Warning: verbs mode with --side-channel-host localhost only works for same-host setups; use a routable IP/hostname for multi-host prefill/decode."
fi

# ── Build flags ──
EP_FLAG=""
if [[ "$EP" -eq 1 ]]; then
    EP_FLAG="--enable-expert-parallel"
fi

QUANT_FLAG=""
[[ -n "$QUANT" ]] && QUANT_FLAG="--quantization $QUANT"

HYBRID_KV_FLAG=""
[[ "$HYBRID_KV" -eq 1 ]] && [[ "$DISAGG" -eq 1 ]] && HYBRID_KV_FLAG="--no-disable-hybrid-kv-cache-manager"

PROFILE_FLAG=""
[[ "$PROFILE" -eq 1 ]] && PROFILE_FLAG='--profiler-config {"profiler":"torch","torch_profiler_dir":"./vllm_profile/","torch_profiler_record_shapes":"true","torch_profiler_with_stack":"true"}'

PREFIX_CACHING_FLAG=""
[[ "$PREFIX_CACHING" -eq 0 ]] && PREFIX_CACHING_FLAG="--no-enable-prefix-caching"

if [[ -n "$NUMA_CONTROL" ]]; then
    export VLLM_NUMA_CONTROL="$NUMA_CONTROL"
fi

DP_FLAG=""
if [[ "$DP" -gt 1 ]]; then
    DP_FLAG="--data-parallel-size $DP"
fi

# ── Disagg-only flags ──
KV_CONFIG_FLAG=""
DISAGG_ENV=()
if [[ "$DISAGG" -eq 1 ]]; then
    DISAGG_UCX_TLS="$UCX_TLS_OVERRIDE"
    if [[ -z "$DISAGG_UCX_TLS" ]]; then
        case "$NIXL_TRANSPORT" in
            ipc)   DISAGG_UCX_TLS="self,tcp,ze_ipc,ze_copy" ;;
            verbs) DISAGG_UCX_TLS="ib,rc,ze_copy" ;;
        esac
    fi
    if [[ "$NIXL_TRANSPORT" == "verbs" ]] && [[ -z "$UCX_NET_DEVICES" ]]; then
        UCX_NET_DEVICES="all"
    fi
    KV_CONFIG_FLAG='--kv-transfer-config {"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"xpu"}'
    DISAGG_ENV=(
        UCX_MEMTYPE_CACHE=0
        UCX_TLS="$DISAGG_UCX_TLS"
        LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/.nixl.mesonpy.libs/plugins:${LD_LIBRARY_PATH:-}"
        VLLM_NIXL_SIDE_CHANNEL_HOST="$SIDE_CHANNEL_HOST"
        VLLM_NIXL_SIDE_CHANNEL_PORT="$SIDE_CHANNEL_PORT"
    )
    if [[ -n "$UCX_NET_DEVICES" ]]; then
        DISAGG_ENV+=(UCX_NET_DEVICES="$UCX_NET_DEVICES")
    fi
fi

# ── Log rotation ──
rotate_log "$LOG_FILE"

# ── Banner in log ──
{
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                     NEW  RUN  STARTING                         ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-63s║\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-63s║\n" "Mode: $LABEL  |  TP=$TP  DP=$DP  EP=$EP"
    printf "║  %-63s║\n" "Model: $(basename "$MODEL")"
    printf "║  %-63s║\n" "Devices: $DEVICE_MASK  |  Port: $PORT"
    [[ -n "$QUANT" ]] && printf "║  %-63s║\n" "Quant: $QUANT  |  Dtype: $DTYPE" \
                      || printf "║  %-63s║\n" "Dtype: $DTYPE"
    printf "║  %-63s║\n" "Attn: $ATTN_BACKEND  |  All2All: $ALL2ALL_BACKEND"
    if [[ "$DISAGG" -eq 1 ]]; then
        printf "║  %-63s║\n" "NIXL transport: $NIXL_TRANSPORT  |  Side host: $SIDE_CHANNEL_HOST"
        printf "║  %-63s║\n" "Resolved UCX_TLS: $DISAGG_UCX_TLS"
        [[ -n "$UCX_NET_DEVICES" ]] && printf "║  %-63s║\n" "UCX_NET_DEVICES: $UCX_NET_DEVICES"
    fi
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
} >> "$LOG_FILE"

# ── Launch ──
echo "Starting $LABEL instance on port $PORT (TP=$TP, DP=$DP, devices=$DEVICE_MASK, all2all=$ALL2ALL_BACKEND)"
[[ -n "$NUMA_CONTROL" ]] && echo "  NUMA binding: $NUMA_CONTROL"
[[ -n "$QUANT" ]] && echo "  Quantization: $QUANT"
if [[ "$DISAGG" -eq 1 ]]; then
    echo "  NIXL transport: $NIXL_TRANSPORT"
    echo "  Resolved UCX_TLS: $DISAGG_UCX_TLS"
    echo "  API host: $SERVE_HOST"
    echo "  Side-channel host: $SIDE_CHANNEL_HOST"
    [[ -n "$UCX_NET_DEVICES" ]] && echo "  UCX_NET_DEVICES: $UCX_NET_DEVICES"
fi

env \
    ZE_AFFINITY_MASK="$DEVICE_MASK" \
    VLLM_USE_V1=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    "${DISAGG_ENV[@]}" \
    vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    $DP_FLAG \
    $EP_FLAG \
    --all2all-backend "$ALL2ALL_BACKEND" \
    --host "$SERVE_HOST" \
    --port "$PORT" \
    --seed 42 \
    --enforce-eager \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    $KV_CONFIG_FLAG \
    $HYBRID_KV_FLAG \
    --max-model-len "$MAX_MODEL_LEN" \
    --block-size "$BLOCK_SIZE" \
    --attention-backend "$ATTN_BACKEND" \
    $PREFIX_CACHING_FLAG \
    $PROFILE_FLAG \
    $QUANT_FLAG >>"$LOG_FILE" 2>&1 &

echo "  PID=$!, log=$LOG_FILE"
