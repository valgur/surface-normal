#pragma once

#include <opencv2/core.hpp>

using namespace cv;

using Plane = cv::Vec4f;

struct CameraParams {
  float f  = 0;
  float cx = 0;
  float cy = 0;
};
extern CameraParams fcxcy;
extern int WINDOWSIZE;
extern float T_threshold;
Plane cvFitPlane(const CvMat *points);
Plane CallFitPlane(const Mat &depth, const int *points, int i, int j);
void search_plane_neighbor(const Mat &img, int i, int j, float threshold, int *result);
bool telldirection(Plane plane, int i, int j, float d);
Mat calplanenormal(const Mat &src);