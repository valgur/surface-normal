#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>

#include "surface_normal/base.hpp"

namespace surface_normal {

namespace py = pybind11;

template <typename T, int channels> ImageView<T, channels>::ImageView(const py::buffer_info &buf) {
  if (!(buf.ndim == 3 && buf.shape[2] == channels) && !(buf.ndim == 2 && channels == 1)) {
    throw std::runtime_error("Incorrect array shape");
  }
  ptr            = reinterpret_cast<uint8_t *>(buf.ptr);
  width          = buf.shape[1];
  height         = buf.shape[0];
  row_stride     = buf.strides[0];
  col_stride     = buf.strides[1];
  channel_stride = buf.ndim == 3 ? buf.strides[2] : sizeof(T);
  if (row_stride <= 0 || col_stride <= 0 || channel_stride <= 0) {
    throw std::runtime_error("Negative-stride arrays are not supported");
  }
}

template <typename T, int channels>
ImageView<T, channels>::ImageView(const py::array &array)
    : ImageView<T, channels>(array.request()) {}

py::array_t<uint8_t> normals_from_depth(const py::array &depth, CameraIntrinsics intrinsics,
                                        int window_size, float max_rel_depth_diff, bool use_cuda) {
  py::buffer_info depth_buf = depth.request();
  if (depth_buf.ndim != 2 && !(depth_buf.ndim == 3 && depth_buf.shape[2] == 1)) {
    throw std::runtime_error("Incorrect depth input array shape");
  };

  py::array_t<uint8_t> normals(
      {static_cast<size_t>(depth_buf.shape[0]), static_cast<size_t>(depth_buf.shape[1]), 3ul});
  ImageView<uint8_t, 3> im_normals(normals);

#define call_with_type(T)                                                                          \
  normals_from_depth(ImageView<const T>(depth_buf), im_normals, intrinsics, window_size,           \
                     max_rel_depth_diff, use_cuda)

  py::dtype depth_dtype = depth.dtype();
  char dtype_kind       = depth_dtype.kind();
  int dtype_bytes       = depth_dtype.itemsize();
  if (dtype_kind == 'f' && dtype_bytes == 4) {
    call_with_type(float);
  } else if (dtype_kind == 'f' && dtype_bytes == 8) {
    call_with_type(double);
  } else if ((dtype_kind == 'u' || dtype_kind == 'B') && dtype_bytes == 1) {
    call_with_type(uint8_t);
  } else if (dtype_kind == 'u' && dtype_bytes == 2) {
    call_with_type(uint16_t);
  } else if (dtype_kind == 'u' && dtype_bytes == 4) {
    call_with_type(uint32_t);
  } else if (dtype_kind == 'u' && dtype_bytes == 8) {
    call_with_type(uint64_t);
  } else if ((dtype_kind == 'i' || dtype_kind == 'b') && dtype_bytes == 1) {
    call_with_type(int8_t);
  } else if (dtype_kind == 'i' && dtype_bytes == 2) {
    call_with_type(int16_t);
  } else if (dtype_kind == 'i' && dtype_bytes == 4) {
    call_with_type(int32_t);
  } else if (dtype_kind == 'i' && dtype_bytes == 8) {
    call_with_type(int64_t);
  } else {
    throw std::runtime_error(std::string("unsupported np.ndarray dtype ") + dtype_kind +
                             std::to_string(dtype_bytes));
  }
#undef call_with_type
  return normals;
}

} // namespace surface_normal