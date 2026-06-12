import sys
import torch
import torch.nn.functional as F
import itertools
import math

import qk_rms_norm_ops


def cosine_similarity_chunked(a, b, chunk_size=4 * 1024 * 1024):
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    n = a_flat.numel()

    dot = 0.0
    a_norm_sq = 0.0
    b_norm_sq = 0.0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        a_chunk = a_flat[start:end].float()
        b_chunk = b_flat[start:end].float()
        dot += torch.sum(a_chunk * b_chunk).item()
        a_norm_sq += torch.sum(a_chunk * a_chunk).item()
        b_norm_sq += torch.sum(b_chunk * b_chunk).item()

    denom = math.sqrt(a_norm_sq) * math.sqrt(b_norm_sq) + 1e-12
    return dot / denom

def reference_qk_rms_norm(token_data, q_norm_weight, k_norm_weight, q_head_num, kv_head_num, qk_head_dim, v_head_dim, eps):
    # in-place norm on specified heads
    q_start = 0
    q_end = q_head_num * qk_head_dim
    q_head_data = token_data[:, q_start:q_end]
    q_head_data = q_head_data.view(-1, q_head_num, qk_head_dim)

    k_start = q_head_num * qk_head_dim
    k_end = k_start + kv_head_num * qk_head_dim
    k_head_data = token_data[:, k_start:k_end]
    k_head_data = k_head_data.view(-1, kv_head_num, qk_head_dim)

    q_head_data = torch.nn.functional.rms_norm(
        q_head_data, 
        normalized_shape=q_head_data.shape[-1:],
        weight=q_norm_weight, 
        eps=eps
    )
    k_head_data = torch.nn.functional.rms_norm(
        k_head_data, 
        normalized_shape=k_head_data.shape[-1:],
        weight=k_norm_weight, 
        eps=eps
    )

    out = token_data.clone()
    out[:, q_start:q_end] = q_head_data.view(-1, q_head_num * qk_head_dim)
    out[:, k_start:k_end] = k_head_data.view(-1, kv_head_num * qk_head_dim) 

    return out

def test_all():
    device = "xpu:0"
    eps = 1e-5
    
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
    
    print("Testing qk_rms_norm accuracy...")
    atol = 1e-3
    cos_threshold = 0.98
    fail_cnt = 0
    mean_diff_sum = 0.0
    case_cnt = 0
    for (q_head_num, kv_head_num, qk_head_dim, v_head_dim), num_tokens in itertools.product(shapes, tokens):
        total_dim = (q_head_num + kv_head_num) * qk_head_dim + kv_head_num * v_head_dim
        
        token_data = torch.randn(num_tokens, total_dim, dtype=torch.bfloat16, device=device) * 0.1
        q_norm_weight = torch.ones(qk_head_dim, dtype=torch.float32, device=device)
        k_norm_weight = torch.ones(qk_head_dim, dtype=torch.float32, device=device)
        
        ref_out = reference_qk_rms_norm(token_data, q_norm_weight, k_norm_weight, q_head_num, kv_head_num, qk_head_dim, v_head_dim, eps)

        # Run kernel in-place on the input tensor to reduce peak memory.
        actual_out = token_data.clone()
        qk_rms_norm_ops.qk_rms_norm_forward(actual_out, q_norm_weight, k_norm_weight, q_head_num, kv_head_num, qk_head_dim, v_head_dim, eps)

        mean_diff = torch.abs(ref_out - actual_out).mean().item()
        cos_sim = cosine_similarity_chunked(ref_out, actual_out)

        mean_diff_sum += mean_diff
        case_cnt += 1

        if mean_diff > atol or cos_sim <= cos_threshold:
            fail_cnt += 1
            print(
                f"FAILED on num_tokens={num_tokens}, q={q_head_num}, kv={kv_head_num}, "
                f"dim={qk_head_dim}: mean_diff {mean_diff}, cos_sim {cos_sim}"
            )
        else:
            print(f"SUCCESS on num_tokens={num_tokens}, q={q_head_num}, kv={kv_head_num}, dim={qk_head_dim}: diff {mean_diff:.3f} cos_sim {cos_sim:.3f}")

        del ref_out, actual_out, token_data, q_norm_weight, k_norm_weight
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
    
    mean_diff_all = mean_diff_sum / case_cnt if case_cnt > 0 else 0.0
    print(f"Mean diff observed: {mean_diff_all}, atol={atol}, cos_threshold={cos_threshold}")
    if fail_cnt > 0:
        print(f"Accuracy test failed, failure count: {fail_cnt}")
        sys.exit(1)

    print("All tests passed successfully.")

if __name__ == "__main__":
    test_all()
