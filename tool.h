#ifndef TOOL
#define TOOL
#include <cassert>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
using namespace cv;
using namespace std;

template <class Type> Type stringToNum(const std::string &str) {
  std::istringstream iss(str);
  Type num;
  iss >> num;
  return num;
}
// string Int_to_String(int n)

// {

// ostringstream stream;

// stream<<n; //n为int类型

// return stream.str();

//}
//-- basic toolbox------
//----------------------
void MkoneDir(const string &stdirname);
void Mat2CvMat(Mat *Input, CvMat *out);
void CvMat2Mat(CvMat *Input, Mat *out);
vector<string> ReadDir(const string &path);
void MkDir(string stdirname);
//根据特定后缀获得文件
std::vector<std::string> get_specific_files(const std::string &path, const std::string &suffix);

// temp test function
Mat closecheck(const Mat &raw);
Mat rawdepth2normal(const Mat &rawdepth, const float *paras);
//-- sparcenormal toolbox------
//-----------------------------
struct CameraParams {
  float f  = 0;
  float cx = 0;
  float cy = 0;
};
extern CameraParams fcxcy;
extern int WINDOWSIZE;
extern float T_threshold;
//生成一个球形物体的深度图用来求normal标称方向
Mat GetaSphere();
Mat sample(const Mat &input);
CameraParams readTxt(const string &file);
void cvFitPlane(const CvMat *points, float *plane);
void CallFitPlane(const Mat &depth, const int *points, int i, int j, float *plane12);
void search_plane_neighbor(const Mat &img, int i, int j, float threshold, int *result);
int telldirection(float *abc, int i, int j, float d);
vector<string> search_working_dir(const string &inputdir);
void get_dir_para(const string &inputdir, float *fcxcy);
Mat calplanenormal(const Mat &src);
Mat caldensenormal(const Mat &rawdept);
//------------------lidar-combine------------------------
//最近邻插值同时生成一张显示最近邻的欧式距离的可信度图
void nearneigbor(const Mat &src, int windowsize, Mat *ress);
int search_neighbor_x(const Mat &img, int i, int j, int circle_size);
int search_neighbor_y(const Mat &img, int i, int j, int circle_size);
Mat sparce_depth2normal(const Mat &input, const float *paras, int circle_size);
//误差欧式距离权重法
void os(const Mat &src, int windowsize, Mat *ress);
#endif
