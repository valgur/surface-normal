#pragma once

#include <cstddef>
#include <type_traits>

#include "surface_normal/cuda_compatibility.hpp"

namespace cv {
class Mat;
}

namespace surface_normal {

// A non-owning wrapper for 2D image arrays, such as cv::Mat and np.ndarray.
// Assumes C-style memory layout.
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

} // namespace surface_normal