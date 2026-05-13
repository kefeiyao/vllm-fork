usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -m, --model MODEL   Model path (default: /host/hf_models/Qwen3-30B-A3B/)
  --pdev DEVICES      Prefill GPU devices, comma-separated (default: 0,1,2,3)
                      Prefill TP is derived from the number of devices
  --ddev DEVICES      Decode GPU devices, comma-separated (default: 4,5,6,7)
                      Decode DP is derived from the number of devices
  --pnuma NUMA        Prefill NUMA control string (default: none)
  --dnuma NUMA        Decode NUMA control string (default: none)
  -h, --help          Show this help message
EOF
  exit 0
}

model=/host/hf_models/Qwen3-30B-A3B/
p_devices=0,1,2,3
d_devices=4,5,6,7
p_numa=""
d_numa=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    -m|--model) model="$2"; shift 2 ;;
    --pdev) p_devices="$2"; shift 2 ;;
    --ddev) d_devices="$2"; shift 2 ;;
    --pnuma) p_numa="$2"; shift 2 ;;
    --dnuma) d_numa="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Derive parallelism from device count
IFS=',' read -ra _p_arr <<< "$p_devices"
p_tp=${#_p_arr[@]}
IFS=',' read -ra _d_arr <<< "$d_devices"
d_dp=${#_d_arr[@]}

# Build optional numa flags
p_numa_flag=""
d_numa_flag=""
[[ -n "$p_numa" ]] && p_numa_flag="-n $p_numa"
[[ -n "$d_numa" ]] && d_numa_flag="-n $d_numa"

bash ./vllm_serve.sh -k
sleep 1
# oneccl correctness WA
#NEOReadDebugKeys=1 EnableImplicitScaling=0 RenderCompressedBuffersEnabled=0
# W8A16
P_A2A_BACKEND="allgather_reducescatter"
#D_A2A_BACKEND="veloci_deepep"
D_A2A_BACKEND="allgather_reducescatter"

#export UCX_MAX_RNDV_RAILS=4
#export UCX_LOG_LEVEL=info
#export UCX_PROTO_INFO=y

bash ./vllm_serve.sh --prefill -m $model -t $p_tp -e -b FLASH_ATTN -d $p_devices -q fp8 -a ${P_A2A_BACKEND} --dtype bfloat16 --gpu-mem-util 0.9 $p_numa_flag --nixl-transport verbs
bash ./vllm_serve.sh --decode -m $model -t 1 --dp $d_dp -e -b FLASH_ATTN -d $d_devices -q fp8 -a ${D_A2A_BACKEND} --dtype bfloat16 --gpu-mem-util 0.9 $d_numa_flag --nixl-transport verbs
bash ./vllm_serve.sh --proxy --wait
