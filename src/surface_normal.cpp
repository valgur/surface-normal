#include <cstdint>

#include "surface_normal/base.hpp"
#include "surface_normal_impl.hpp"

namespace surface_normal {

void normals_from_depth_cpu(const ImageView<const float> &depth, ImageView<uint8_t, 3> &normals,
                            CameraIntrinsics intrinsics, int window_size,
                            float max_rel_depth_diff) {
  for (int row = 0; row < depth.height; row++) {
    for (int col = 0; col < depth.width; col++) {
      depth_to_normals_rgb_inner(depth, normals, intrinsics, window_size / 2, max_rel_depth_diff,
                                 row, col);
    }
  }
}

} // namespace surface_normal