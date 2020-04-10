#pragma once

#include <cstddef>
#include <type_traits>

#include "surface_normal/cuda_compatibility.hpp"

namespace cv {
class Mat;
}

namespace surface_normal {

struct CameraIntrinsics {
  float f;
  float cx;
  float cy;
};

template <typename T> class ImageView {
public:
  T *data;
  int width{};
  int height{};
  int channels = 1;
  int step     = width * channels;

  __hdi__ int step_bytes() const { return step * sizeof(T); }

  __hdi__ size_t size() const { return height * step; }
  __hdi__ size_t size_bytes() const { return height * step_bytes(); }

  __hdi__ T *row(int row) const { return data + row * step; }

  __hdi__ T &at(int row_, int col, int channel = 0) const {
    return row(row_)[col * channels + channel];
  }

  __hdi__ uint8_t *data_raw() const {
    return reinterpret_cast<uint8_t *>(const_cast<typename std::remove_const<T>::type *>(data));
  }

  ImageView() = default;
  explicit ImageView(const cv::Mat &m);
};

void normals_from_depth_cpu(const ImageView<const float> &depth, ImageView<uint8_t> &normals,
                            CameraIntrinsics intrinsics, int window_size = 15,
                            float max_rel_depth_diff = 0.1);
extern "C" {
void normals_from_depth_cuda(const ImageView<const float> &depth, ImageView<uint8_t> &normals,
                             CameraIntrinsics intrinsics, int window_size = 15,
                             float max_rel_depth_diff = 0.1);
}
} // namespace surface_normal