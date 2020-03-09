#include <iostream>
#include <map>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "tool.h"

std::vector<bool> find_neighbors(const Mat &img, int i, int j, float threshold,
                                 size_t window_size) {
  int cols = img.cols;
  int rows = img.rows;
  std::vector<bool> plane_points_mask(window_size * window_size, false);
  float center_depth = img.at<float>(i, j);
  for (int idx = 0; idx < window_size; idx++) {
    for (int idy = 0; idy < window_size; idy++) {
      int rx = i - int(window_size / 2) + idx;
      int ry = j - int(window_size / 2) + idy;
      if (rx >= rows || ry >= cols) {
        continue;
      }
      if (img.at<float>(rx, ry) == 0.0) {
        continue;
      }
      if (abs(img.at<float>(rx, ry) - center_depth) <= threshold * center_depth) {
        plane_points_mask[idx * window_size + idy] = true;
      }
    }
  }
  return plane_points_mask;
}

// Ax+by+cz=D
Plane call_fit_plane(const Mat &depth, const std::vector<bool> &points_mask, int i, int j,
                     CameraParams intrinsics, int window_size) {
  float f_inv = 1.f / intrinsics.f;
  float cx    = intrinsics.cx;
  float cy    = intrinsics.cy;
  std::vector<Vec3f> points_vec;
  for (int num_point = 0; num_point < window_size * window_size; num_point++) {
    if (!points_mask[num_point])
      continue;
    int point_i = num_point / window_size;
    int point_j = num_point % window_size;
    point_i += i - window_size / 2;
    point_j += j - window_size / 2;
    float x = (point_j - cx) * depth.at<float>(point_i, point_j) * f_inv;
    float y = (point_i - cy) * depth.at<float>(point_i, point_j) * f_inv;
    float z = depth.at<float>(point_i, point_j);
    points_vec.emplace_back(x, y, z);
  }
  if (points_vec.size() < 3) {
    return Plane(-1, -1, -1, -1);
  }
  cv::Mat points_mat(points_vec.size(), 3, CV_32F);
  for (int ii = 0; ii < points_vec.size(); ++ii) {
    points_mat.at<float>(ii, 0) = points_vec[ii][0];
    points_mat.at<float>(ii, 1) = points_vec[ii][1];
    points_mat.at<float>(ii, 2) = points_vec[ii][2];
  }
  Plane plane12 = fit_plane(points_mat);
  if (telldirection(plane12, i, j, depth.at<float>(i, j), intrinsics)) {
    plane12[0] = -plane12[0];
    plane12[1] = -plane12[1];
    plane12[2] = -plane12[2];
  }
  return plane12;
}

Plane fit_plane(const Mat &points) {
  // Estimate geometric centroid.
  int nrows = points.rows;
  int ncols = 3;
  int type  = points.type();
  Vec3f centroid(0, 0, 0);
  for (int c = 0; c < ncols; c++) {
    for (int r = 0; r < nrows; r++) {
      centroid[c] += points.at<float>(r, c);
    }
  }
  centroid /= static_cast<float>(nrows);
  // Subtract geometric centroid from each point.
  Mat points2(nrows, ncols, type);
  for (int r = 0; r < nrows; r++) {
    for (int c = 0; c < ncols; c++) {
      points2.at<float>(r, c) = points.at<float>(r, c) - centroid[c];
    }
  }
  // Evaluate SVD of covariance matrix.
  Mat A;
  gemm(points2, points, 1, noArray(), 0, A, CV_GEMM_A_T);
  SVD svd(A, SVD::MODIFY_A);
  // Assign plane coefficients by singular std::vector corresponding to smallest
  // singular value.
  Plane plane;
  plane[ncols] = 0;
  for (int c = 0; c < ncols; c++) {
    plane[c] = svd.vt.at<float>(ncols - 1, c);
    plane[ncols] += plane[c] * centroid[c];
  }
  return plane;
}

bool telldirection(Plane plane, int i, int j, float d, CameraParams intrinsics) {
  float f_inv  = 1.f / intrinsics.f;
  float cx     = intrinsics.cx;
  float cy     = intrinsics.cy;
  float x      = (j - cx) * d * f_inv;
  float y      = (i - cy) * d * f_inv;
  float z      = d;
  Vec3f cor    = Vec3f(0 - x, 0 - y, 0 - z);
  Vec3f normal = Vec3f(plane[0], plane[1], plane[2]);
  return cor.dot(normal) >= 0;
}

Mat normals_from_depth(const Mat &src, CameraParams intrinsics, int window_size,
                       float rel_dist_threshold) {
  Mat normals = Mat::zeros(src.size(), CV_32FC3);
  src.convertTo(src, CV_32FC1);
  src *= 1.0;
  int cols = src.cols;
  int rows = src.rows;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      // for kitti and nyud test
      if (src.at<float>(i, j) == 0) {
        continue;
      }
      // for:nyud train
      //  if(src.at<float>(i,j)<=4000.0)continue;

      auto plane_points_mask = find_neighbors(src, i, j, rel_dist_threshold, window_size);
      Plane plane12 = call_fit_plane(src, plane_points_mask, i, j, intrinsics, window_size);
      Vec3f d       = Vec3f(plane12[0], plane12[1], plane12[2]);
      Vec3f n       = normalize(d);
      normals.at<Vec3f>(i, j) = n;
    }
  }
  Mat res = Mat::zeros(src.size(), CV_32FC3);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      res.at<Vec3f>(i, j)[0] = -normals.at<Vec3f>(i, j)[0];
      res.at<Vec3f>(i, j)[2] = -normals.at<Vec3f>(i, j)[1];
      res.at<Vec3f>(i, j)[1] = -normals.at<Vec3f>(i, j)[2];
    }
  }
  normals.release();

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (res.at<Vec3f>(i, j)[0] == 0 && res.at<Vec3f>(i, j)[1] == 0 && res.at<Vec3f>(i, j)[2] == 0)
        continue;
      res.at<Vec3f>(i, j)[0] += 1;
      res.at<Vec3f>(i, j)[1] += 1;
      res.at<Vec3f>(i, j)[2] += 1;
    }
  }

  res *= 127.5;
  res.convertTo(res, CV_8UC3);
  return res;
}

int main() {
  CameraParams intrinsics{};
  intrinsics.f             = 721.5377;
  intrinsics.cx            = 596.5593;
  intrinsics.cy            = 149.854;
  int window_size          = 15;
  float rel_dist_threshold = 0.1;

  cv::Mat depth = cv::imread("gt.png", cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
  depth.convertTo(depth, CV_32F);
  cv::Mat res = normals_from_depth(depth, intrinsics, window_size, rel_dist_threshold);
  cvtColor(res, res, COLOR_BGR2RGB);
  cv::imwrite("gt_out.png", res);

  return 0;
}
