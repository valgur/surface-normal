#include <array>
#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

struct CameraParams {
  float f  = 0;
  float cx = 0;
  float cy = 0;
};

using Plane = std::array<float, 4>;

CameraParams fcxcy;  // SET THE CAMERA PARAMETERS  F CX CY BEFORE USE calplanenormal.
int WINDOWSIZE = 15; // SET SEARCH WINDOWSIZE(SUGGEST 15) BEFORE USE calplanenormal.
float T_threshold;   // SET THE threshold (SUGGEST 0.1-0.2)BEFORE USE
                     // calplanenormal.

void cvFitPlane(const CvMat *points, float *plane);
void CallFitPlane(const Mat &depth, int *points, int i, int j, float *plane12);
void search_plane_neighbor(const Mat &img, int i, int j, float threshold, int *result);
bool telldirection(float *abc, int i, int j, float d);
Mat calplanenormal(const Mat &src);

// Ax+by+cz=D
void CallFitPlane(const Mat &depth, int *points, int i, int j, float *plane12) {
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
    plane12[0] *= -1;
    plane12[1] *= -1;
    plane12[2] *= -1;
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
  // Assign plane coefficients by singular vector corresponding to smallest
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
bool telldirection(float *abc, int i, int j, float d) {
  float f  = fcxcy.f;
  float cx = fcxcy.cx;
  float cy = fcxcy.cy;
  float x  = (j - cx) * d * 1.0 / f;
  float y  = (i - cy) * d * 1.0 / f;
  float z  = d;
  // Vec3f camera_center=Vec3f(cx,cy,0);
  Vec3f cor     = Vec3f(0 - x, 0 - y, 0 - z);
  Vec3f abcline = Vec3f(abc);
  float corner  = cor.dot(abcline);
  //  float corner =(cx-x)*abc[0]+(cy-y) *abc[1]+(0-z)*abc[2];
  return corner >= 0;
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
      normals.at<Vec3f>(i, j) = normalize(Vec3f(plane12));
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
  // set parameters here:fcxcy.f=0;fcxcy.cx=0;fcxcy.cy=0;
  std::string INPUT_FILE_NAME = "in_xyz";
  std::string OUTPUT_NAME     = "out_xyz";
  Mat src                     = imread(INPUT_FILE_NAME, CV_LOAD_IMAGE_ANYDEPTH);
  Mat res                     = calplanenormal(src);
  imwrite(OUTPUT_NAME, res);
  return 0;
}