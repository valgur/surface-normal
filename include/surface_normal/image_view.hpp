#pragma once

#include <cstddef> // size_t

#include "surface_normal/cuda_compatibility.hpp"

namespace cv {
class Mat;
}
namespace pybind11 {
class array;
struct buffer_info;
} // namespace pybind11

namespace surface_normal {

// A non-owning wrapper for 2D image arrays, such as cv::Mat and np.ndarray.
// Assumes C-style memory layout.
template <typename T, int channels = 1> class ImageView {
public:
  uint8_t *ptr{};
  int width{};
  int height{};
  int channel_stride = sizeof(T);
  int col_stride     = channels * channel_stride;
  int row_stride     = width * col_stride;

  __hdi__ size_t size() const { return height * width * channels; }
  __hdi__ size_t size_bytes() const { return height * row_stride; }

  __hdi__ T &at(int row, int col, int channel = 0) const {
    return *reinterpret_cast<T *>(ptr + row * row_stride + col * col_stride +
                                  channel * channel_stride);
  }

  ImageView() = default;
  ImageView(uint8_t *ptr, int width, int height) : ptr(ptr), width(width), height(height) {}
  explicit ImageView(const cv::Mat &m);
  explicit ImageView(const pybind11::buffer_info &buf);
  explicit ImageView(const pybind11::array &array);
};

} // namespace surface_normal