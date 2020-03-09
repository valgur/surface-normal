#pragma once

#include <opencv2/core.hpp>

using namespace cv;

struct CameraParams {
  float f  = 0;
  float cx = 0;
  float cy = 0;
};
extern CameraParams fcxcy;
extern int WINDOWSIZE;
extern float T_threshold;
void cvFitPlane(const CvMat *points, float *plane);
void CallFitPlane(const Mat &depth, const int *points, int i, int j, float *plane12);
void search_plane_neighbor(const Mat &img, int i, int j, float threshold, int *result);
int telldirection(float *abc, int i, int j, float d);
Mat calplanenormal(const Mat &src);