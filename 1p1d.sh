bash ./prefill_ipc.sh -k
model=/host/hf_models/Qwen3-30B-A3B/
# oneccl correctness WA
#NEOReadDebugKeys=1 EnableImplicitScaling=0 RenderCompressedBuffersEnabled=0
# W8A16
bash ./prefill_ipc.sh -m $model -t 4 -e -b FLASH_ATTN  -d 0,1,2,3 -q fp8 -a "allgather_reducescatter" --dtype bfloat16 -n "0-15:0;16-31:0;64-79:0;80-95:0"
VLLM_VELOCI_DEEPEP_USE_A2A=1 bash ./decode_ipc.sh -m  $model -t 1 --dp 4 -e -b FLASH_ATTN  -d 4,5,6,7 -q fp8 -a "veloci_deepep" --dtype bfloat16 -n "32-47:1;48-63:1;96-111:1;112-127:1"
#bash ./decode_ipc.sh -m  $model -t 1 --dp 4 -e -b FLASH_ATTN  -d 4,5,6,7 -q fp8 -a "allgather_reducescatter" --dtype bfloat16 -n "32-47:1;48-63:1;96-111:1;112-127:1"

# W16A16
#bash ./prefill_ipc.sh -m $model -t 4 -e -b FLASH_ATTN  -d 0,1,2,3 --dtype bfloat16 -n "0-15:0;16-31:0;64-79:0;80-95:0"
#bash ./decode_ipc.sh -m  $model -t 1 --dp 4 -e -b FLASH_ATTN  -d 4,5,6,7 --dtype bfloat16 -n "32-47:1;48-63:1;96-111:1;112-127:1"
#

# P: TP4EP4, D: TP2EP2
#bash ./prefill_ipc.sh -m $model -t 4 -e -b FLASH_ATTN  -d 0,1,2,3 -q fp8 --dtype bfloat16 -n "0-15:0;16-31:0;64-79:0;80-95:0"
#bash ./decode_ipc.sh -m  $model -t 2 --dp 1 -e -b FLASH_ATTN  -d 4,5,6,7 -q fp8 --dtype bfloat16 -n "32-47:1;48-63:1;96-111:1;112-127:1"
