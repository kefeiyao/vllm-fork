// Head RMS Norm SYCL extension for xpu-perf.
// ESIMD implementation.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

using namespace sycl::ext::intel::esimd;

using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;

constexpr int SIMD = 16;

inline float esimd_scalar_rsqrt(float x) {
  simd<float, 1> v(x);
  return rsqrt(v)[0];
}

struct SyclKerConfigBase {};

#define __SYCL_KER_CONFIG_CONVENTION__ SyclKerConfigBase
#define SYCL_REQD_SUB_GROUP_SIZE(SIZE) [[sycl::reqd_sub_group_size(SIZE)]]



struct StrideSelection {
  int64_t stride_dim0;
  int64_t head_start_offset;
  int64_t head_num;
  int64_t head_dim;
};

// Merged ESIMD kernel for both Q and K normalization
// One work group processes N_T tokens (default N_T=2), processing both Q and K heads
template <typename T, int CHUNK, int N_T = 2>
void launch_esimd_kernel_qk_merged(
    int num_tokens,
    int N,
    float eps,
    T* X,
    const float* q_gamma,
    const float* k_gamma,
    StrideSelection q_selection,
    StrideSelection k_selection) {
  auto& queue = c10::xpu::getCurrentXPUStream().queue();

  constexpr int wg_size = 32;
  const int q_head_num = q_selection.head_num;
  const int k_head_num = k_selection.head_num;
  const int total_heads = q_head_num + k_head_num;
  
  // Calculate number of workgroups: each workgroup handles N_T tokens
  const int num_wg = (num_tokens + N_T - 1) / N_T;

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_wg * wg_size), sycl::range<1>(wg_size)),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL [[intel::kernel_args_restrict]] {
          const int wg_id = item.get_group(0);
          const int lid = item.get_local_id(0);

          // Each workgroup processes N_T tokens
          const int token_start = wg_id * N_T;
          const int token_end = (token_start + N_T < num_tokens) ? (token_start + N_T) : num_tokens;

          // Process N_T tokens assigned to this workgroup
          for (int token_idx = token_start; token_idx < token_end; ++token_idx) {
            // Load gamma weights for Q
            simd<float, CHUNK> gf = block_load<float, CHUNK>(q_gamma);

            // Process Q heads for this token
            T* q_base = X + token_idx * q_selection.stride_dim0 + q_selection.head_start_offset;
            for (int head_idx = lid; head_idx < q_head_num; head_idx += wg_size) {
              T* row_data = q_base + head_idx * q_selection.head_dim;

              simd<T, CHUNK> v = block_load<T, CHUNK>(row_data);
              simd<float, CHUNK> vf = simd<float, CHUNK>(v);

              float var_sum = reduce<float>(vf * vf, std::plus<>{});
              vf = vf * gf;
              const float denom = esimd_scalar_rsqrt(var_sum / static_cast<float>(N) + eps);
              vf = vf * denom;

              simd<T, CHUNK> out = simd<T, CHUNK>(vf);
              block_store<T, CHUNK>(row_data, out);
            }

            // Load gamma weights for K
            gf = block_load<float, CHUNK>(k_gamma);

            // Process K heads for this token
            T* k_base = X + token_idx * k_selection.stride_dim0 + k_selection.head_start_offset;
            for (int head_idx = lid; head_idx < k_head_num; head_idx += wg_size) {
              T* row_data = k_base + head_idx * k_selection.head_dim;

              simd<T, CHUNK> v = block_load<T, CHUNK>(row_data);
              simd<float, CHUNK> vf = simd<float, CHUNK>(v);

              float var_sum = reduce<float>(vf * vf, std::plus<>{});
              vf = vf * gf;
              const float denom = esimd_scalar_rsqrt(var_sum / static_cast<float>(N) + eps);
              vf = vf * denom;

              simd<T, CHUNK> out = simd<T, CHUNK>(vf);
              block_store<T, CHUNK>(row_data, out);
            }
          }
        });
  });
}

// Merged launcher for both Q and K normalization
// N_T: number of tokens per workgroup (default=2)
template <typename T, int N_T = 1>
void launch_qk_rms_norm_esimd_merged(
    int N,
    int num_tokens,
    float eps,
    T* X,
    const float* q_gamma,
    const float* k_gamma,
    StrideSelection q_selection,
    StrideSelection k_selection) {
    switch (N) {
      case 8:
        launch_esimd_kernel_qk_merged<T, 8, N_T>(num_tokens, N, eps, X, q_gamma, k_gamma, q_selection, k_selection);
        break;
      case 16:
        launch_esimd_kernel_qk_merged<T, 16, N_T>(num_tokens, N, eps, X, q_gamma, k_gamma, q_selection, k_selection);
        break;
      case 32:
        launch_esimd_kernel_qk_merged<T, 32, N_T>(num_tokens, N, eps, X, q_gamma, k_gamma, q_selection, k_selection);
        break;
      case 64:
        launch_esimd_kernel_qk_merged<T, 64, N_T>(num_tokens, N, eps, X, q_gamma, k_gamma, q_selection, k_selection);
        break;
      case 96:
        launch_esimd_kernel_qk_merged<T, 96, N_T>(num_tokens, N, eps, X, q_gamma, k_gamma, q_selection, k_selection);
        break;
      case 128:
        launch_esimd_kernel_qk_merged<T, 128, N_T>(num_tokens, N, eps, X, q_gamma, k_gamma, q_selection, k_selection);
        break;
      case 256:
        launch_esimd_kernel_qk_merged<T, 256, N_T>(num_tokens, N, eps, X, q_gamma, k_gamma, q_selection, k_selection);
        break;
      case 512:
        launch_esimd_kernel_qk_merged<T, 512, N_T>(num_tokens, N, eps, X, q_gamma, k_gamma, q_selection, k_selection);
        break;
      case 1024:
        launch_esimd_kernel_qk_merged<T, 1024, N_T>(num_tokens, N, eps, X, q_gamma, k_gamma, q_selection, k_selection);
        break;
      default:
        TORCH_CHECK(false, "Unsupported head_dim for merged kernel: ", N);
    }
}

torch::Tensor qk_rms_norm_forward(
    torch::Tensor token_data,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& k_norm_weight,
    int64_t q_head_num,
    int64_t kv_head_num,
    int64_t qk_head_dim,
    int64_t v_head_dim,
    double eps) {

  TORCH_CHECK(token_data.is_xpu(), "token_data must be an XPU tensor");
  TORCH_CHECK(q_norm_weight.is_xpu(), "q_norm_weight must be an XPU tensor");
  TORCH_CHECK(k_norm_weight.is_xpu(), "k_norm_weight must be an XPU tensor");
  TORCH_CHECK(token_data.is_contiguous(), "token_data must be contiguous");
  TORCH_CHECK(q_norm_weight.is_contiguous(), "q_norm_weight must be contiguous");
  TORCH_CHECK(k_norm_weight.is_contiguous(), "k_norm_weight must be contiguous");

  int64_t num_tokens = token_data.size(0);
  int64_t total_dim = token_data.size(1);
  int64_t expected_total_dim = (q_head_num + kv_head_num) * qk_head_dim + kv_head_num * v_head_dim;
  TORCH_CHECK(total_dim == expected_total_dim, "total_dim mismatch");

  // Q heads selection
  StrideSelection q_selection;
  q_selection.stride_dim0 = total_dim;
  q_selection.head_start_offset = 0;
  q_selection.head_num = q_head_num;
  q_selection.head_dim = qk_head_dim;
  int64_t q_M = num_tokens * q_head_num;

  // K heads selection
  StrideSelection k_selection;
  k_selection.stride_dim0 = total_dim;
  k_selection.head_start_offset = q_head_num * qk_head_dim;
  k_selection.head_num = kv_head_num;
  k_selection.head_dim = qk_head_dim;
  int64_t k_M = num_tokens * kv_head_num;

  // Use merged kernel to process both Q and K in a single kernel launch
  if (token_data.scalar_type() == at::ScalarType::Half) {
    const fp16* x_ptr = reinterpret_cast<const fp16*>(token_data.const_data_ptr<c10::Half>());
    const float* q_gamma_ptr = q_norm_weight.const_data_ptr<float>();
    const float* k_gamma_ptr = k_norm_weight.const_data_ptr<float>();
    
    launch_qk_rms_norm_esimd_merged<fp16>(
        static_cast<int>(qk_head_dim),
        static_cast<int>(num_tokens),
        static_cast<float>(eps),
        const_cast<fp16*>(x_ptr),
        q_gamma_ptr,
        k_gamma_ptr,
        q_selection,
        k_selection);
  } else if (token_data.scalar_type() == at::ScalarType::BFloat16) {
    const bf16* x_ptr = reinterpret_cast<const bf16*>(token_data.const_data_ptr<c10::BFloat16>());
    const float* q_gamma_ptr = q_norm_weight.const_data_ptr<float>();
    const float* k_gamma_ptr = k_norm_weight.const_data_ptr<float>();
    
    launch_qk_rms_norm_esimd_merged<bf16>(
        static_cast<int>(qk_head_dim),
        static_cast<int>(num_tokens),
        static_cast<float>(eps),
        const_cast<bf16*>(x_ptr),
        q_gamma_ptr,
        k_gamma_ptr,
        q_selection,
        k_selection);
  } else if (token_data.scalar_type() == at::ScalarType::Float) {
    const float* x_ptr = token_data.const_data_ptr<float>();
    const float* q_gamma_ptr = q_norm_weight.const_data_ptr<float>();
    const float* k_gamma_ptr = k_norm_weight.const_data_ptr<float>();
    
    launch_qk_rms_norm_esimd_merged<float>(
        static_cast<int>(qk_head_dim),
        static_cast<int>(num_tokens),
        static_cast<float>(eps),
        const_cast<float*>(x_ptr),
        q_gamma_ptr,
        k_gamma_ptr,
        q_selection,
        k_selection);
  } else {
    TORCH_CHECK(false, "Unsupported input dtype. Use float16, bfloat16, or float32.");
  }

  return token_data;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "qk_rms_norm_forward",
      &qk_rms_norm_forward,
      "Head RMSNorm forward (SYCL extension, inplace)");
}