import torch
import time
import math
import random
import json
import itertools, sys
from functools import reduce
from operator import mul

try:
    import qk_rms_norm_ops
except ImportError:
    qk_rms_norm_ops = None

class OpTensorInfo(object):
    def __init__(self, shape, dtype, creator=None, device=None):
        self.shape = shape
        self.dtype = dtype
        self.creator = creator
        self.device = device

def calc_tensor_size(tensor_info):
    if isinstance(tensor_info, OpTensorInfo):
        return reduce(mul, tensor_info.shape, 1) * torch.empty([], dtype=tensor_info.dtype).element_size()
    return 0

class DummyOp:
    def __init__(self, num_tokens, q_head_num, kv_head_num, qk_head_dim, v_head_dim, dtype, provider="qk_rms_norm"):
        self.num_tokens = num_tokens
        self.q_head_num = q_head_num
        self.kv_head_num = kv_head_num
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.dtype = dtype
        self.provider = provider
        self.eps = 1e-5
        
        self.norm_dim = (self.q_head_num + self.kv_head_num) * self.qk_head_dim
        self.not_norm_dim = self.kv_head_num * self.v_head_dim
        self.total_dim = self.norm_dim + self.not_norm_dim

        self.input_tensor_info = {
            "token_data": OpTensorInfo(
                shape=[self.num_tokens, self.total_dim],
                dtype=self.dtype,
            ),
            "q_norm_weight": OpTensorInfo(
                shape=[self.qk_head_dim, ],
                dtype=torch.float32,
            ),
            "k_norm_weight": OpTensorInfo(
                shape=[self.qk_head_dim, ],
                dtype=torch.float32,
            ),
        }
        self.output_tensor_info = {}

        self.input_tensor_size = sum(
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        )
        self.output_tensor_size = 0
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        data_bytes = calc_tensor_size(self.input_tensor_info["token_data"]) / self.total_dim * self.norm_dim
        q_weight_bytes = calc_tensor_size(self.input_tensor_info["q_norm_weight"])
        k_weight_bytes = calc_tensor_size(self.input_tensor_info["k_norm_weight"])

        self.read_bytes = data_bytes + q_weight_bytes + k_weight_bytes
        self.write_bytes = data_bytes
        self.io_bytes = self.read_bytes + self.write_bytes

        self.calc_flops = 0

    def create_tensors(self, count):
        tensors = []
        for _ in range(count):
            token_data = torch.randn(self.num_tokens, self.total_dim, dtype=self.dtype, device='xpu')
            q_norm_weight = torch.ones(self.qk_head_dim, dtype=torch.float32, device='xpu')
            k_norm_weight = torch.ones(self.qk_head_dim, dtype=torch.float32, device='xpu')
            tensors.append({"token_data": token_data, "q_norm_weight": q_norm_weight, "k_norm_weight": k_norm_weight})
        return tensors

    def core_run(self, tensor_mapping):
        token_data = tensor_mapping["token_data"]
        q_norm_weight = tensor_mapping["q_norm_weight"]
        k_norm_weight = tensor_mapping["k_norm_weight"]

        if self.provider == "qk_rms_norm":
            if qk_rms_norm_ops is None:
                raise RuntimeError("qk_rms_norm_ops is not available; build it first")
            
            qk_rms_norm_ops.qk_rms_norm_forward(
                token_data, q_norm_weight, k_norm_weight, self.q_head_num, self.kv_head_num, self.qk_head_dim, self.v_head_dim, self.eps
            )
            return token_data
        else:
            raise RuntimeError("Unsupported provider: {}".format(self.provider))

    def summary(self, latency_us):
        target_dict = {}
        if latency_us > 0:
            target_dict['provider'] = self.provider
            target_dict['latency(us)'] = round(latency_us, 3)
            target_dict['read_bytes(B)'] = self.read_bytes
            target_dict['write_bytes(B)'] = self.write_bytes
            target_dict['io_bytes(B)'] = self.io_bytes
            target_dict['mem_bw(GB/s)'] = round(self.io_bytes / latency_us / 1e3, 3)
            target_dict['calc_flops'] = self.calc_flops
            target_dict['calc_flops_power(tflops)'] = round(self.calc_flops / latency_us / 1e6, 3)
            target_dict['calc_mem_ratio'] = round(self.calc_flops / self.io_bytes, 3) if self.io_bytes != 0 else 0
            target_dict['kernels'] = [] # empty kernel list
        return target_dict

def core_perf(op, warmup, prefer, tensor_list):
    for i in range(warmup):
        op.core_run(tensor_list[i % len(tensor_list)])
    torch.xpu.synchronize()

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)

    torch.xpu.synchronize()
    start_event.record()
    for i in range(prefer):
        op.core_run(tensor_list[i % len(tensor_list)])
    end_event.record()
    end_event.synchronize()
    torch.xpu.synchronize()

    return start_event.elapsed_time(end_event) * 1e3 / prefer

def run_perf(op):
    torch.xpu.empty_cache()
    avail = torch.xpu.get_device_properties(0).total_memory - torch.xpu.memory_allocated(0)
    assume_avail = int(avail * 0.9)
    assume_cache = 1024**3
    
    tensor_size = op.tensor_size
    max_iters = 10000
    
    max_data_cnt = 1
    if tensor_size > assume_avail:
        raise RuntimeError('OOM')
    elif 2 * tensor_size > assume_avail:
        max_data_cnt = 1
    elif tensor_size > assume_cache:
        max_data_cnt = 2
    else:
        max_data_cnt = min(math.floor(max(assume_avail, assume_cache) / tensor_size), math.floor(assume_cache / tensor_size))
        
    tensor_list = op.create_tensors(max_data_cnt)
    random.shuffle(tensor_list)
    
    lat = core_perf(op, 10, 10, tensor_list)
    prefer_iters = min(max(int(5000000 / lat), 2), max_iters)

    pass # time.sleep(0.2)
    final_lat = core_perf(op, 10, prefer_iters, tensor_list)
    
    del tensor_list
    torch.xpu.empty_cache()
    pass # time.sleep(0.5)
    
    return op.summary(final_lat)

def main():
    providers = []
    if qk_rms_norm_ops is not None:
        providers.append("qk_rms_norm")
    else:
        print( "qk_rms_norm_ops not found, skipping benchmark.")
        return

    print( 'Benchmarking qk_rms_norm kernel...')
    
    shapes = [
        [64, 8, 128, 128],
        [80, 8, 128, 128],
        [32, 4, 128, 128],
        [40, 4, 128, 128],
        [16, 2, 128, 128],
        [20, 2, 128, 128],
        [8, 1, 128, 128],
        [16, 1, 128, 128]
    ]
    tokens = [16, 64, 192, 384, 896, 1024, 1792, 4096, 5120, 10240, 32768]

    for provider in providers:
        for dtype_name in ['bfloat16']:
            dtype = getattr(torch, dtype_name)
            for (q_head_num, kv_head_num, qk_head_dim, v_head_dim), num_tokens in itertools.product(shapes, tokens):
                op = DummyOp(
                    num_tokens, q_head_num, kv_head_num, qk_head_dim, v_head_dim, dtype,
                    provider=provider)
                try:
                    res = run_perf(op)
                    print( 
                        f"{{'provider': '{provider}', 'arg_type': 'llm', "
                        f"'dtype': '{dtype_name}', "
                        f"'num_tokens': {num_tokens}, 'q_head_num.kv_head_num.qk_head_dim.v_head_dim': "
                        f"[{q_head_num}, {kv_head_num}, {qk_head_dim}, {v_head_dim}]}}"
                    )
                    print( json.dumps(res, indent=4))
                except Exception as e:
                    print( 
                        f"ERROR provider={provider} dtype={dtype_name} "
                        f"config=({num_tokens}, {q_head_num}, {kv_head_num}, {qk_head_dim}, {v_head_dim}): {e}"
                    )

if __name__ == '__main__':
    main()
