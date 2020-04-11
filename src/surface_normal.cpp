#include <cstdint>

#include "surface_normal/base.hpp"
#include "surface_normal_impl.hpp"

namespace surface_normal {

template <typename T>
void normals_from_depth_cpu(const ImageView<const T> &depth, ImageView<uint8_t, 3> &normals,
                            CameraIntrinsics intrinsics, int window_size,
                            float max_rel_depth_diff) {
  std::memset(normals.ptr, 0, normals.size_bytes());
  for (int row = 0; row < depth.height; row++) {
    for (int col = 0; col < depth.width; col++) {
      depth_to_normals_rgb_inner(depth, normals, intrinsics, window_size / 2, max_rel_depth_diff,
                                 row, col);
    }
  }
}

#define instantiate(T)                                                                             \
  template void normals_from_depth_cpu(                                                            \
      const ImageView<const T> &depth, ImageView<uint8_t, 3> &normals,                             \
      CameraIntrinsics intrinsics, int window_size, float max_rel_depth_diff)

instantiate(float);
instantiate(double);
instantiate(uint8_t);
instantiate(uint16_t);
instantiate(uint32_t);
instantiate(uint64_t);
instantiate(int8_t);
instantiate(int16_t);
instantiate(int32_t);
instantiate(int64_t);

} // namespace surface_normal