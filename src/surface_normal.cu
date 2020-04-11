#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <opencv2/core.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "surface_normal/base.hpp"
#include "surface_normal_impl.hpp"

#define SAFE_CALL(call, msg) _safe_cuda_call((call), (msg), __FILE__, __LINE__)
static inline void _safe_cuda_call(cudaError err, const char *msg, const char *file_name,
                                   const int line_number) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name,
            line_number, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

namespace surface_normal {

// Note: objects like depth and normals cannot be passed by reference to a kernel function
template <typename T>
__global__ void d_depth_to_normals(const ImageView<const T> depth, ImageView<uint8_t, 3> normals,
                                   CameraIntrinsics intrinsics, int r, float max_rel_depth_diff) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= depth.width || row >= depth.height)
    return;
  depth_to_normals_rgb_inner(depth, normals, intrinsics, r, max_rel_depth_diff, row, col);
}

template <typename T>
void normals_from_depth_cuda(const ImageView<const T> &depth, ImageView<uint8_t, 3> &normals,
                             CameraIntrinsics intrinsics, int window_size,
                             float max_rel_depth_diff) {
  ImageView<const T> d_depth(depth);
  ImageView<uint8_t, 3> d_normals(normals);

  // Allocate device memory
  SAFE_CALL(cudaMalloc(&d_depth.data, d_depth.size_bytes()), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc(&d_normals.data, normals.size_bytes()), "CUDA Malloc Failed");

  // Copy data from input image to device memory
  SAFE_CALL(cudaMemcpy(d_depth.data, depth.data, d_depth.size_bytes(), cudaMemcpyHostToDevice),
            "CUDA Memcpy Host To Device Failed");
  SAFE_CALL(cudaMemset(d_normals.data, 0, d_normals.size_bytes()), "CUDA memset normals to 0");

  // Specify a reasonable block size
  const dim3 block(16, 16);

  // Calculate grid size to cover the whole image
  const dim3 grid((depth.width + block.x - 1) / block.x, (depth.height + block.y - 1) / block.y);

  // Launch the kernel
  d_depth_to_normals<<<grid, block>>>(d_depth, d_normals, intrinsics, window_size / 2,
                                      max_rel_depth_diff);

  // Synchronize to check for any kernel launch errors
  SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

  // Copy back data from device memory to OpenCV output image
  SAFE_CALL(
      cudaMemcpy(normals.data, d_normals.data, d_normals.size_bytes(), cudaMemcpyDeviceToHost),
      "CUDA Memcpy Host To Device Failed");

  // Free the device memory
  SAFE_CALL(cudaFree(d_depth.data), "CUDA Free Failed");
  SAFE_CALL(cudaFree(d_normals.data), "CUDA Free Failed");
}

#define instantiate(T)                                                                             \
  template void normals_from_depth_cuda(                                                           \
      const ImageView<const T> &depth, ImageView<uint8_t, 3> &normals,                             \
      CameraIntrinsics intrinsics, int window_size, float max_rel_depth_diff)

instantiate(float);
instantiate(double);
instantiate(uint8_t);
instantiate(uint16_t);
instantiate(int32_t);
instantiate(int64_t);

} // namespace surface_normal