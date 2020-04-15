#pragma once

#include <cmath>
#include <cstdint>

#include <Eigen/Core>

#include "surface_normal/base.hpp"
#include "surface_normal/cuda_compatibility.hpp"
#include "svd3_cuda.hpp"

namespace surface_normal {

__host__ __device__ __forceinline__ uint8_t f2b(float x) {
  return static_cast<uint8_t>(127.5 * (1 - x));
}

template <typename T>
__host__ __device__ __forceinline__ void
depth_to_normals_rgb_inner(const ImageView<const T> &depth, ImageView<uint8_t, 3> &normals,
                           const CameraIntrinsics &intrinsics, int radius, float max_rel_depth_diff,
                           int center_row, int center_col) {
  float center_depth = depth.at(center_row, center_col);
  if (center_depth == 0) {
    return;
  }

  float f_inv = 1.f / intrinsics.f;
  float cx    = intrinsics.cx;
  float cy    = intrinsics.cy;
  Eigen::Vector3f mid{(center_col - cx) * center_depth * f_inv,
                      (center_row - cy) * center_depth * f_inv, center_depth};
  int n                          = 0;
  Eigen::Vector3f centroid       = Eigen::Vector3f::Zero();
  Eigen::Matrix3f outer_prod_sum = Eigen::Matrix3f::Zero();
  for (int i = -radius; i <= radius; i++) {
    for (int j = -radius; j <= radius; j++) {
      int x = center_col + j;
      int y = center_row + i;

      if (x < 0 || x >= depth.width || y < 0 || y >= depth.height) {
        continue;
      }

      float z = depth.at(y, x);
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
  Eigen::Matrix3f cov = (outer_prod_sum - n * centroid * centroid.transpose()) / (n - 1);

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

  normals.at(center_row, center_col, 0) = f2b(normal(0));
  normals.at(center_row, center_col, 1) = f2b(normal(2));
  normals.at(center_row, center_col, 2) = f2b(normal(1));
}
} // namespace surface_normal
