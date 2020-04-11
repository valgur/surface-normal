#pragma once

#include <stdexcept>
#include <tuple>

#include "surface_normal/image_view.hpp"

namespace surface_normal {

// For Python bindings
using CameraIntrinsicsTuple = std::tuple<double, double, double>;

struct CameraIntrinsics {
  float f;
  float cx;
  float cy;

  CameraIntrinsics(CameraIntrinsicsTuple t) {
    f  = std::get<0>(t);
    cx = std::get<1>(t);
    cy = std::get<2>(t);
  }
};

template <typename T>
extern void normals_from_depth_cpu(const ImageView<const T> &depth, ImageView<uint8_t, 3> &normals,
                                   CameraIntrinsics intrinsics, int window_size = 15,
                                   float max_rel_depth_diff = 0.1);
template <typename T>
extern void normals_from_depth_cuda(const ImageView<const T> &depth, ImageView<uint8_t, 3> &normals,
                                    CameraIntrinsics intrinsics, int window_size = 15,
                                    float max_rel_depth_diff = 0.1);

template <typename T>
void normals_from_depth(const ImageView<const T> &depth, ImageView<uint8_t, 3> &normals,
                        CameraIntrinsics intrinsics, int window_size = 15,
                        float max_rel_depth_diff = 0.1, bool use_cuda = USE_CUDA) {
  if (use_cuda) {
#ifdef WITH_CUDA
    normals_from_depth_cuda(depth, normals, intrinsics, window_size, max_rel_depth_diff);
#else
    throw std::runtime_error("module was built without CUDA support, but use_cuda=True");
#endif
  } else {
    normals_from_depth_cpu(depth, normals, intrinsics, window_size, max_rel_depth_diff);
  }
}

} // namespace surface_normal