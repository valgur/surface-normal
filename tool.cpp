#include <iostream>
#include <map>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "tool.h"

CameraParams fcxcy;
// 15 0.1 for kitti lidar, 7 0.1 for nyu
int WINDOWSIZE    = 15;
float T_threshold = 0.1;

const std::map<const std::string, const CameraParams> date_to_intrinsics{
    {"2011_09_26", {721.5377, 596.5593, 149.854}},  {"2011_09_28", {707.0493, 604.0814, 162.5066}},
    {"2011_09_29", {718.3351, 600.3891, 159.5122}}, {"2011_09_30", {707.0912, 601.8873, 165.1104}},
    {"2011_10_03", {718.856, 607.1928, 161.2157}},
};

void search_plane_neighbor(const Mat &img, int i, int j, float threshold, int *result) {
  int cols = img.cols;
  int rows = img.rows;
  for (int ii = 0; ii < WINDOWSIZE * WINDOWSIZE; ii++) {
    result[ii] = 0;
  }
  float center_depth = img.at<float>(i, j);
  for (int idx = 0; idx < WINDOWSIZE; idx++) {
    for (int idy = 0; idy < WINDOWSIZE; idy++) {
      int rx = i - int(WINDOWSIZE / 2) + idx;
      int ry = j - int(WINDOWSIZE / 2) + idy;
      if (rx >= rows || ry >= cols) {
        continue;
      }
      if (img.at<float>(rx, ry) == 0.0) {
        continue;
      }
      if (abs(img.at<float>(rx, ry) - center_depth) <= T_threshold * center_depth) {
        result[idx * WINDOWSIZE + idy] = 1;
      }
    }
  }
}

// Ax+by+cz=D
void CallFitPlane(const Mat &depth, const int *points, int i, int j, float *plane12) {
  float f  = fcxcy.f;
  float cx = fcxcy.cx;
  float cy = fcxcy.cy;
  std::vector<float> X_vector;
  std::vector<float> Y_vector;
  std::vector<float> Z_vector;
  for (int num_point = 0; num_point < WINDOWSIZE * WINDOWSIZE; num_point++) {
    if (points[num_point] == 1) { // search 已经处理了边界,此处不需要再处理了
      int point_i, point_j;
      point_i = floor(num_point / WINDOWSIZE);
      point_j = num_point - (point_i * WINDOWSIZE);
      point_i += i - int(WINDOWSIZE / 2);
      point_j += j - int(WINDOWSIZE / 2);
      float x = (point_j - cx) * depth.at<float>(point_i, point_j) * 1.0 / f;
      float y = (point_i - cy) * depth.at<float>(point_i, point_j) * 1.0 / f;
      float z = depth.at<float>(point_i, point_j);
      X_vector.push_back(x);
      Y_vector.push_back(y);
      Z_vector.push_back(z);
    }
  }
  CvMat *points_mat = cvCreateMat(X_vector.size(), 3, CV_32FC1); //定义用来存储需要拟合点的矩阵
  if (X_vector.size() < 3) {
    plane12[0] = -1;
    plane12[1] = -1;
    plane12[2] = -1;
    plane12[3] = -1;
    return;
  }
  for (int ii = 0; ii < X_vector.size(); ++ii) {
    points_mat->data.fl[ii * 3 + 0] = X_vector[ii]; //矩阵的值进行初始化   X的坐标值
    points_mat->data.fl[ii * 3 + 1] = Y_vector[ii]; //  Y的坐标值
    points_mat->data.fl[ii * 3 + 2] = Z_vector[ii]; //
  }
  // float plane12[4] = { 0 };//定义用来储存平面参数的数组
  cvFitPlane(points_mat, plane12); //调用方程
  if (telldirection(plane12, i, j, depth.at<float>(i, j))) {
    plane12[0] = -plane12[0];
    plane12[1] = -plane12[1];
    plane12[2] = -plane12[2];
  }
  X_vector.clear();
  Y_vector.clear();
  Z_vector.clear();
  cvReleaseMat(&points_mat);
}

void cvFitPlane(const CvMat *points, float *plane) {
  // Estimate geometric centroid.
  int nrows       = points->rows;
  int ncols       = points->cols;
  int type        = points->type;
  CvMat *centroid = cvCreateMat(1, ncols, type);
  cvSet(centroid, cvScalar(0));
  for (int c = 0; c < ncols; c++) {
    for (int r = 0; r < nrows; r++) {
      centroid->data.fl[c] += points->data.fl[ncols * r + c];
    }
    centroid->data.fl[c] /= nrows;
  }
  // Subtract geometric centroid from each point.
  CvMat *points2 = cvCreateMat(nrows, ncols, type);
  for (int r = 0; r < nrows; r++) {
    for (int c = 0; c < ncols; c++) {
      points2->data.fl[ncols * r + c] = points->data.fl[ncols * r + c] - centroid->data.fl[c];
    }
  }
  // Evaluate SVD of covariance matrix.
  CvMat *A = cvCreateMat(ncols, ncols, type);
  CvMat *W = cvCreateMat(ncols, ncols, type);
  CvMat *V = cvCreateMat(ncols, ncols, type);
  cvGEMM(points2, points, 1, nullptr, 0, A, CV_GEMM_A_T);
  cvSVD(A, W, nullptr, V, CV_SVD_V_T);
  // Assign plane coefficients by singular std::vector corresponding to smallest
  // singular value.
  plane[ncols] = 0;
  for (int c = 0; c < ncols; c++) {
    plane[c] = V->data.fl[ncols * (ncols - 1) + c];
    plane[ncols] += plane[c] * centroid->data.fl[c];
  }
  // Release allocated resources.
  cvReleaseMat(&centroid);
  cvReleaseMat(&points2);
  cvReleaseMat(&A);
  cvReleaseMat(&W);
  cvReleaseMat(&V);
}

int telldirection(float *abc, int i, int j, float d) {
  float f  = fcxcy.f;
  float cx = fcxcy.cx;
  float cy = fcxcy.cy;
  float x  = (j - cx) * d * 1.0 / f;
  float y  = (i - cy) * d * 1.0 / f;
  float z  = d;
  // Vec3f camera_center=Vec3f(cx,cy,0);
  Vec3f cor     = Vec3f(0 - x, 0 - y, 0 - z);
  Vec3f abcline = Vec3f(abc[0], abc[1], abc[2]);
  float corner  = cor.dot(abcline);
  //  float corner =(cx-x)*abc[0]+(cy-y) *abc[1]+(0-z)*abc[2];
  if (corner >= 0) {
    return 1;
  }
  return 0;
}

Mat calplanenormal(const Mat &src) {
  Mat normals = Mat::zeros(src.size(), CV_32FC3);
  src.convertTo(src, CV_32FC1);
  src *= 1.0;
  int cols = src.cols;
  int rows = src.rows;
  //  int plane_points[WINDOWSIZE*WINDOWSIZE]={0};
  int *plane_points = new int[WINDOWSIZE * WINDOWSIZE];
  float *plane12    = new float[4];
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      // for kitti and nyud test
      if (src.at<float>(i, j) == 0.0) {
        continue;
      }
      // for:nyud train
      //  if(src.at<float>(i,j)<=4000.0)continue;

      search_plane_neighbor(src, i, j, 15.0, plane_points);
      CallFitPlane(src, plane_points, i, j, plane12);
      Vec3f d                 = Vec3f(plane12[0], plane12[1], plane12[2]);
      Vec3f n                 = normalize(d);
      normals.at<Vec3f>(i, j) = n;
    }
  }
  Mat res = Mat::zeros(src.size(), CV_32FC3);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      res.at<Vec3f>(i, j)[0] = -1.0 * normals.at<Vec3f>(i, j)[0];
      res.at<Vec3f>(i, j)[2] = -1.0 * normals.at<Vec3f>(i, j)[1];
      res.at<Vec3f>(i, j)[1] = -1.0 * normals.at<Vec3f>(i, j)[2];
    }
  }

  delete[] plane12;
  delete[] plane_points;
  normals.release();
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (!(res.at<Vec3f>(i, j)[0] == 0 && res.at<Vec3f>(i, j)[1] == 0 &&
            res.at<Vec3f>(i, j)[2] == 0)) {
        res.at<Vec3f>(i, j)[0] += 1.0;
        res.at<Vec3f>(i, j)[2] += 1.0;
        res.at<Vec3f>(i, j)[1] += 1.0;
      }
    }
  }

  res = res * 127.5;
  res.convertTo(res, CV_8UC3);
  cvtColor(res, res, COLOR_BGR2RGB);
  return res;
}

int main() {
  fcxcy.f     = 721.5377;
  fcxcy.cx    = 596.5593;
  fcxcy.cy    = 149.854;
  cv::Mat src = cv::imread("gt.png", cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
  src.convertTo(src, CV_32F);
  std::cout << "cols: " << src.cols << std::endl;
  std::cout << "rows: " << src.rows << std::endl;
  cv::Mat res = calplanenormal(src);
  std::cout << "cols: " << res.cols << std::endl;
  std::cout << "rows: " << res.rows << std::endl;
  std::cout << "type: " << res.type() << std::endl;
  std::cout << "depth: " << res.depth() << std::endl;
  std::cout << "channels: " << res.channels() << std::endl;
  cv::imwrite("gt_out.png", res);

  return 0;
}
