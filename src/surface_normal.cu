#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "surface_normal.hpp"
#include "svd3_cuda.hpp"

static inline void _safe_cuda_call(cudaError err, const char *msg, const char *file_name,
                                   const int line_number) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name,
            line_number, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
#define SAFE_CALL(call, msg) _safe_cuda_call((call), (msg), __FILE__, __LINE__)

__device__ __forceinline__ uint8_t f2b(float x) { return static_cast<uint8_t>(127.5 * (1 - x)); }

__global__ void d_depth_to_normals_rgb(const float *depth, uint8_t *normals, int width, int height,
                                       int input_step, int output_step, CameraIntrinsics intrinsics,
                                       int r, float max_rel_depth_diff) {
  int x_center = blockIdx.x * blockDim.x + threadIdx.x;
  int y_center = blockIdx.y * blockDim.y + threadIdx.y;

  if (x_center >= width || y_center >= height)
    return;

  float center_depth = depth[y_center * input_step + x_center];
  if (center_depth == 0) {
    return;
  }

  float f_inv = 1.f / intrinsics.f;
  float cx    = intrinsics.cx;
  float cy    = intrinsics.cy;
  Eigen::Vector3f mid{(x_center - cx) * center_depth * f_inv,
                      (y_center - cy) * center_depth * f_inv, center_depth};
  int n                          = 0;
  Eigen::Vector3f centroid       = Eigen::Vector3f::Zero();
  Eigen::Matrix3f outer_prod_sum = Eigen::Matrix3f::Zero();
  for (int i = -r; i <= r; i++) {
    for (int j = -r; j <= r; j++) {
      int x = x_center + j;
      int y = y_center + i;

      if (x >= width || y >= height) {
        continue;
      }

      float z = depth[y * input_step + x];
      if (z == 0 || std::abs(z - center_depth) > max_rel_depth_diff * center_depth) {
        continue;
      }

      Eigen::Vector3f p{(x - cx) * z * f_inv, (y - cy) * z * f_inv, z};
      p -= mid; // subtract midpoint for improved numeric stability in outer product
      centroid += p;
      // '* 1' to suppress
      // warning: calling a __host__ function from a __host__ __device__ function is not allowed
      outer_prod_sum += p * p.transpose() * 1;
      n++;
    }
  }

  if (n < 3)
    return;

  centroid /= n;
  Eigen::Matrix3f cov = (outer_prod_sum - centroid * centroid.transpose() * n) / (n - 1);

  Eigen::Matrix3f U, V;
  Eigen::Vector3f S;

  svd(cov(0, 0), cov(0, 1), cov(0, 2), cov(1, 0), cov(1, 1), cov(1, 2), cov(2, 0), cov(2, 1),
      cov(2, 2),                                                                       // cov
      U(0, 0), U(0, 1), U(0, 2), U(1, 0), U(1, 1), U(1, 2), U(2, 0), U(2, 1), U(2, 2), // output U
      S(0), S(1), S(2),                                                                // output S
      V(0, 0), V(0, 1), V(0, 2), V(1, 0), V(1, 1), V(1, 2), V(2, 0), V(2, 1), V(2, 2)  // output V
  );
  Eigen::Vector3f normal = V.col(2).normalized();

  if (mid.dot(normal) < 0) {
    normal *= -1;
  }

  normals[y_center * output_step + 3 * x_center + 0] = f2b(normal(0));
  normals[y_center * output_step + 3 * x_center + 1] = f2b(normal(2));
  normals[y_center * output_step + 3 * x_center + 2] = f2b(normal(1));
}

cv::Mat3b normals_from_depth_cuda(const cv::Mat1f &depth, CameraIntrinsics intrinsics,
                                  int window_size, float max_rel_depth_diff) {
  cv::Mat output(depth.size(), CV_8UC3);

  // Calculate total number of bytes of input and output image
  const int input_bytes  = depth.step * depth.rows;
  const int output_bytes = output.step * output.rows;

  uint8_t *d_input;
  uint8_t *d_output;

  // Allocate device memory
  SAFE_CALL(cudaMalloc(&d_input, input_bytes), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc(&d_output, output_bytes), "CUDA Malloc Failed");

  // Copy data from OpenCV input image to device memory
  SAFE_CALL(cudaMemcpy(d_input, depth.ptr(), input_bytes, cudaMemcpyHostToDevice),
            "CUDA Memcpy Host To Device Failed");
  SAFE_CALL(cudaMemset(d_output, 0, output_bytes), "CUDA memset output to 0");

  // Specify a reasonable block size
  const dim3 block(16, 16);

  // Calculate grid size to cover the whole image
  const dim3 grid((depth.cols + block.x - 1) / block.x, (depth.rows + block.y - 1) / block.y);

  // Launch the kernel
  d_depth_to_normals_rgb<<<grid, block>>>(reinterpret_cast<const float *>(d_input), d_output,
                                          depth.cols, depth.rows, depth.step / sizeof(float),
                                          output.step / sizeof(uint8_t), intrinsics,
                                          window_size / 2, max_rel_depth_diff);

  // Synchronize to check for any kernel launch errors
  SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

  // Copy back data from destination device meory to OpenCV output image
  SAFE_CALL(cudaMemcpy(output.ptr(), d_output, output_bytes, cudaMemcpyDeviceToHost),
            "CUDA Memcpy Host To Device Failed");

  // Free the device memory
  SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
  SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");

  return output;
}