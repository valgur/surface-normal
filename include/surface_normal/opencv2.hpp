#pragma once

#include <cstdint>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "surface_normal/base.hpp"

namespace surface_normal {

template <typename T, int channels> ImageView<T, channels>::ImageView(const cv::Mat &m) {
  data           = const_cast<uint8_t *>(m.ptr());
  width          = m.cols;
  height         = m.rows;
  row_step_bytes = m.step;
  assert(channels == m.channels());
}

cv::Mat3f normals_from_depth(const cv::Mat &depth, CameraIntrinsics intrinsics,
                             int window_size = 15, float max_rel_depth_diff = 0.1,
                             bool use_cuda = USE_CUDA) {
  cv::Mat3b normals = cv::Mat::zeros(depth.size(), CV_8UC3);
  ImageView<uint8_t, 3> im_normals(normals);
  switch (depth.type()) {
  case (CV_32F):
    normals_from_depth(ImageView<const float>(depth), im_normals, intrinsics, window_size,
                       max_rel_depth_diff, use_cuda);
    break;
  case (CV_64F):
    normals_from_depth(ImageView<const double>(depth), im_normals, intrinsics, window_size,
                       max_rel_depth_diff, use_cuda);
    break;
  case (CV_8U):
    normals_from_depth(ImageView<const uint8_t>(depth), im_normals, intrinsics, window_size,
                       max_rel_depth_diff, use_cuda);
    break;
  case (CV_8S):
    normals_from_depth(ImageView<const int8_t>(depth), im_normals, intrinsics, window_size,
                       max_rel_depth_diff, use_cuda);
    break;
  case (CV_16U):
    normals_from_depth(ImageView<const uint16_t>(depth), im_normals, intrinsics, window_size,
                       max_rel_depth_diff, use_cuda);
    break;
  case (CV_32S):
    normals_from_depth(ImageView<const int32_t>(depth), im_normals, intrinsics, window_size,
                       max_rel_depth_diff, use_cuda);
    break;
  default:
    throw std::runtime_error("unsupported cv::Mat type " + std::to_string(depth.type()));
  }
  return normals;
}

void normals_from_depth_imgfile(const std::string &depth_in_path,
                                const std::string &normals_out_path,
                                const CameraIntrinsics &intrinsics, int window_size,
                                float max_rel_depth_diff, bool use_cuda) {
  cv::Mat depth = cv::imread(depth_in_path, cv::IMREAD_UNCHANGED);
  if (depth.size().area() == 0) {
    throw std::runtime_error("Empty image");
  }
  if (depth.channels() != 1) {
    throw std::runtime_error("Not a single-channel depth image. Image has " +
                             std::to_string(depth.channels()) + " channels.");
  }
  cv::Mat3f normals_rgb =
      normals_from_depth(depth, intrinsics, window_size, max_rel_depth_diff, use_cuda);
  cvtColor(normals_rgb, normals_rgb, cv::COLOR_RGB2BGR);
  imwrite(normals_out_path, normals_rgb);
}

} // namespace surface_normal