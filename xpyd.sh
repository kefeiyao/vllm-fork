bash ./vllm_serve.sh -k
sleep 1
model=/host/hf_models/Qwen3-30B-A3B/
# oneccl correctness WA
#NEOReadDebugKeys=1 EnableImplicitScaling=0 RenderCompressedBuffersEnabled=0
# W8A16
# --no-prefix-caching
P_A2A_BACKEND="allgather_reducescatter"
#D_A2A_BACKEND="veloci_deepep"
D_A2A_BACKEND="allgather_reducescatter"
bash ./vllm_serve.sh --prefill -m $model -t 4 -e -b FLASH_ATTN  -d 0,1,2,3 -q fp8 -a ${P_A2A_BACKEND} --dtype bfloat16 -n "0-15:0;16-31:0;64-79:0;80-95:0"
bash ./vllm_serve.sh --decode -m  $model -t 1 --dp 4 -e -b FLASH_ATTN  -d 4,5,6,7 -q fp8 -a ${D_A2A_BACKEND} --dtype bfloat16 -n "32-47:1;48-63:1;96-111:1;112-127:1"
bash ./vllm_serve.sh --proxy --wait
