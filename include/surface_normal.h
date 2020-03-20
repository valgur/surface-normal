#pragma once

#include <vector>

#include <opencv2/core.hpp>

using Plane = cv::Vec4f;

struct CameraIntrinsics {
  float f;
  float cx;
  float cy;
};

cv::Mat3f normals_from_depth(const cv::Mat &depth, CameraIntrinsics intrinsics, int window_size,
                             float rel_dist_threshold);

cv::Mat3b normals_to_rgb(const cv::Mat3f &normals);

cv::Mat1f get_surrounding_points(const cv::Mat &depth, int i, int j, CameraIntrinsics intrinsics,
                                 size_t window_size, float threshold);

cv::Vec3f fit_plane(const cv::Mat &points);
