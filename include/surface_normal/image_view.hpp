#pragma once

#include <cstddef> // size_t

#include "surface_normal/cuda_compatibility.hpp"

namespace cv {
class Mat;
}

namespace surface_normal {

// A non-owning wrapper for 2D image arrays, such as cv::Mat and np.ndarray.
// Assumes C-style memory layout.
template <typename T, int channels = 1> class ImageView {
public:
  uint8_t *data{};
  int width{};
  int height{};
  int row_step_bytes = width * channels * sizeof(T);

  __hdi__ size_t size() const { return height * width * channels; }
  __hdi__ size_t size_bytes() const { return height * row_step_bytes; }

  __hdi__ T *row(int row) const { return reinterpret_cast<T *>(data + row * row_step_bytes); }

  __hdi__ T &at(int row_, int col, int channel = 0) const {
    return row(row_)[col * channels + channel];
  }

  ImageView() = default;
  ImageView(uint8_t *data, int width, int height) : data(data), width(width), height(height) {}
  explicit ImageView(const cv::Mat &m);
};

} // namespace surface_normal