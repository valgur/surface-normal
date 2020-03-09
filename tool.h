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

Mat calplanenormal(const Mat &src);

std::vector<bool> find_neighbors(const Mat &img, int i, int j, float threshold, size_t window_size);

Plane call_fit_plane(const Mat &depth, const std::vector<bool> &points_mask, int i, int j,
                     CameraParams intrinsics, int window_size);

Plane fit_plane(const Mat &points);

bool telldirection(Plane plane, int i, int j, float d, CameraParams intrinsics);
