#pragma once

#include <vector>

#include <opencv2/core.hpp>

using namespace cv;

using Plane = cv::Vec4f;

struct CameraParams {
  float f;
  float cx;
  float cy;
};

Mat3f normals_from_depth(const Mat &depth, CameraParams intrinsics, int window_size,
                         float rel_dist_threshold);

Mat3b normals_to_rgb(const Mat &normals);

Mat1f get_surrounding_points(const Mat &depth, int i, int j, CameraParams intrinsics,
                             size_t window_size, float threshold);

Plane fit_plane(const Mat &points);
