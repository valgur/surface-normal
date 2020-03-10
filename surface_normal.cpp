#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "surface_normal.h"

Mat1f get_surrounding_points(const Mat &depth, int i, int j, CameraParams intrinsics,
                             size_t window_size, float threshold) {
  float f_inv        = 1.f / intrinsics.f;
  float cx           = intrinsics.cx;
  float cy           = intrinsics.cy;
  float center_depth = depth.at<float>(i, j);
  Mat1f points(window_size * window_size, 3);
  int count = 0;
  for (int idx = 0; idx < window_size; idx++) {
    for (int idy = 0; idy < window_size; idy++) {
      int row = i - int(window_size / 2) + idx;
      int col = j - int(window_size / 2) + idy;
      if (row >= depth.rows || col >= depth.cols) {
        continue;
      }
      float z = depth.at<float>(row, col);
      if (z == 0) {
        continue;
      }
      if (abs(z - center_depth) > threshold * center_depth) {
        continue;
      }
      float x = (col - cx) * z * f_inv;
      float y = (row - cy) * z * f_inv;

      points.at<float>(count, 0) = x;
      points.at<float>(count, 1) = y;
      points.at<float>(count, 2) = z;
      count++;
    }
  }
  return points(Rect(0, 0, 3, count));
}

// Ax+by+cz=D
Vec3f fit_plane(const Mat &points) {
  constexpr int ncols = 3;
  Mat cov, centroid;
  calcCovarMatrix(points, cov, centroid, CV_COVAR_ROWS | CV_COVAR_NORMAL, CV_32F);
  SVD svd(cov, SVD::MODIFY_A);
  // Assign plane coefficients by the singular vector corresponding to the smallest
  // singular value.
  Vec3f normal = normalize(Vec3f(svd.vt.row(ncols - 1)));
  // Plane plane;
  // plane[ncols] = 0;
  // for (int c = 0; c < ncols; c++) {
  //   plane[c] = svd.vt.at<float>(ncols - 1, c);
  //   plane[ncols] += plane[c] * centroid.at<float>(0, c);
  // }
  return normal;
}

Mat3f normals_from_depth(const Mat &depth, CameraParams intrinsics, int window_size,
                         float rel_dist_threshold) {
  Mat3f normals = Mat::zeros(depth.size(), CV_32FC3);
  for (int i = 0; i < depth.rows; i++) {
    for (int j = 0; j < depth.cols; j++) {
      if (depth.at<float>(i, j) == 0) {
        continue;
      }

      Mat1f points =
          get_surrounding_points(depth, i, j, intrinsics, window_size, rel_dist_threshold);

      if (points.rows < 3) {
        continue;
      }

      Vec3f normal = fit_plane(points);
      Vec3f cor    = Vec3f(j - intrinsics.cx, i - intrinsics.cy, intrinsics.f);
      if (cor.dot(normal) < 0) {
        normal *= -1;
      }
      normals.at<Vec3f>(i, j) = normal;
    }
  }
  return normals;
}

constexpr uint8_t f2b(float x) { return static_cast<uint8_t>(127.5 * (1 - x)); }

Mat3b normals_to_rgb(const Mat &normals) {
  Mat3b res = Mat::zeros(normals.size(), CV_8UC3);
  for (int i = 0; i < normals.rows; i++) {
    for (int j = 0; j < normals.cols; j++) {
      if (normals.at<Vec3f>(i, j)[0] == 0 && normals.at<Vec3f>(i, j)[1] == 0 &&
          normals.at<Vec3f>(i, j)[2] == 0)
        continue;
      res.at<Vec3b>(i, j)[0] = f2b(normals.at<Vec3f>(i, j)[0]);
      res.at<Vec3b>(i, j)[2] = f2b(normals.at<Vec3f>(i, j)[1]);
      res.at<Vec3b>(i, j)[1] = f2b(normals.at<Vec3f>(i, j)[2]);
    }
  }
  return res;
}

int main() {
  CameraParams intrinsics{};
  intrinsics.f             = 721.5377;
  intrinsics.cx            = 596.5593;
  intrinsics.cy            = 149.854;
  int window_size          = 15;
  float rel_dist_threshold = 0.1;

  Mat depth = cv::imread("gt.png", cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
  depth.convertTo(depth, CV_32F);
  Mat3f normals     = normals_from_depth(depth, intrinsics, window_size, rel_dist_threshold);
  Mat3b normals_rgb = normals_to_rgb(normals);
  cvtColor(normals_rgb, normals_rgb, COLOR_BGR2RGB);
  cv::imwrite("gt_out.png", normals_rgb);

  return 0;
}