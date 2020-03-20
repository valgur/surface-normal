#include <string>
#include <tuple>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <pybind11/pybind11.h>

#include "surface_normal.h"

namespace py = pybind11;

using CameraIntrinsicsTuple = std::tuple<double, double, double>;

void normals_from_depth_wrapper(const std::string &in_img_path, const std::string &out_img_path,
                                CameraIntrinsicsTuple intrinsics_tuple, int window_size,
                                float rel_dist_threshold = 0.1) {
  CameraIntrinsics intrinsics{};
  intrinsics.f  = std::get<0>(intrinsics_tuple);
  intrinsics.cx = std::get<1>(intrinsics_tuple);
  intrinsics.cy = std::get<2>(intrinsics_tuple);
  cv::Mat depth = cv::imread(in_img_path, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
  depth.convertTo(depth, CV_32F);
  cv::Mat3f normals     = normals_from_depth(depth, intrinsics, window_size, rel_dist_threshold);
  cv::Mat3b normals_rgb = normals_to_rgb(normals);
  cvtColor(normals_rgb, normals_rgb, cv::COLOR_RGB2BGR);
  imwrite(out_img_path, normals_rgb);
}

PYBIND11_MODULE(surface_normal, m) {
  m.doc() = "";
  m.def("normals_from_depth", &normals_from_depth_wrapper, py::arg("in_img_path"),
        py::arg("out_img_path"), py::arg("intrinsics"), py::arg("window_size") = 15,
        py::arg("rel_dist_threshold") = 0.1);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}