#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#include <verification_metadata.h>

namespace {
// One block per virtual batch row. Compact indptr is computed from GPU base
// length; unused capacity stays outside indptr and is never attended to.
__global__ void metadata_kernel(
    const int64_t* base, int64_t* positions, int32_t* ap,
    int32_t* ai, int32_t* ao, int32_t* cp, int32_t* ci, int32_t* co,
    int batch, int rows, int page_size) {
  const int token = blockIdx.x % rows, b = blockIdx.x / rows;
  const int start = static_cast<int>(*base);
  const int final_pages = (start + rows + page_size - 1) / page_size;
  const int length = start + token + 1;
  const int pages = (length + page_size - 1) / page_size;
  int per_batch = 0, prior = 0;
  for (int t = 0; t < rows; ++t) {
    const int p = (start + t + page_size) / page_size;
    per_batch += p;
    if (t < token) prior += p;
  }
  const int offset = b * per_batch + prior;
  if (threadIdx.x == 0) {
    cp[b * rows + token] = offset;
    co[b * rows + token] = (length - 1) % page_size + 1;
    if (b == 0) positions[token] = start + token;
    if (token == 0) {
      ap[b] = b * final_pages;
      ao[b] = (start + rows - 1) % page_size + 1;
    }
    if (b == batch - 1 && token == rows - 1) {
      cp[batch * rows] = batch * per_batch;
      ap[batch] = batch * final_pages;
    }
  }
  for (int p = threadIdx.x; p < pages; p += blockDim.x)
    ci[offset + p] = p * batch + b;
  if (token == 0)
    for (int p = threadIdx.x; p < final_pages; p += blockDim.x)
      ai[b * final_pages + p] = p * batch + b;
}
}

void verification_metadata(
    torch::Tensor base, torch::Tensor positions,
    torch::Tensor append_indptr, torch::Tensor append_indices,
    torch::Tensor append_offsets, torch::Tensor causal_indptr,
    torch::Tensor causal_indices, torch::Tensor causal_offsets,
    int64_t batch, int64_t page_size, int64_t max_pages) {
  TORCH_CHECK(base.is_cuda() && base.scalar_type() == torch::kInt64 &&
              base.numel() == 1 && base.is_contiguous(), "invalid base length");
  const c10::hip::HIPGuard guard(static_cast<c10::DeviceIndex>(base.get_device()));
  TORCH_CHECK(positions.device() == base.device() && positions.is_contiguous()
              && positions.scalar_type() == torch::kInt64,
              "positions must be contiguous GPU int64");
  const int rows = positions.numel();
  TORCH_CHECK(batch > 0 && rows > 0 && rows <= 16 && page_size > 0 &&
              max_pages > 0, "invalid metadata dimensions");
  for (const auto& t : {append_indptr, append_indices, append_offsets,
                       causal_indptr, causal_indices, causal_offsets})
    TORCH_CHECK(t.device() == base.device() && t.is_contiguous() &&
                t.scalar_type() == torch::kInt32, "metadata must be GPU int32");
  TORCH_CHECK(append_indptr.numel() == batch + 1 &&
              append_indices.numel() >= batch * max_pages &&
              append_offsets.numel() == batch &&
              causal_indptr.numel() == batch * rows + 1 &&
              causal_indices.numel() >= batch * rows * max_pages &&
              causal_offsets.numel() == batch * rows, "metadata capacity mismatch");
  hipLaunchKernelGGL(metadata_kernel, dim3(batch * rows), dim3(64), 0,
      at::hip::getCurrentHIPStream().stream(), base.data_ptr<int64_t>(),
      positions.data_ptr<int64_t>(), append_indptr.data_ptr<int32_t>(),
      append_indices.data_ptr<int32_t>(), append_offsets.data_ptr<int32_t>(),
      causal_indptr.data_ptr<int32_t>(), causal_indices.data_ptr<int32_t>(),
      causal_offsets.data_ptr<int32_t>(), batch, rows, page_size);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "metadata launch failed");
}
