#include <string>
#include <tuple>

#include <pybind11/pybind11.h>

#include "surface_normal/opencv2.hpp"

namespace py = pybind11;
using namespace surface_normal;

using CameraIntrinsicsTuple = std::tuple<double, double, double>;

void normals_from_depth_wrapper(const std::string &depth_in_path,
                                const std::string &normals_out_path,
                                CameraIntrinsicsTuple intrinsics_tuple, int window_size,
                                float max_rel_depth_diff, bool use_cuda) {
  CameraIntrinsics intrinsics{};
  intrinsics.f  = std::get<0>(intrinsics_tuple);
  intrinsics.cx = std::get<1>(intrinsics_tuple);
  intrinsics.cy = std::get<2>(intrinsics_tuple);
  normals_from_depth_imgfile(depth_in_path, normals_out_path, intrinsics, window_size,
                             max_rel_depth_diff, use_cuda);
}

PYBIND11_MODULE(surface_normal, m) {
  m.doc() = "";
  m.def("normals_from_depth", &normals_from_depth_wrapper, py::arg("depth_in_path"),
        py::arg("normals_out_path"), py::arg("intrinsics"), py::arg("window_size") = 15,
        py::arg("max_rel_depth_diff") = 0.1, py::arg("use_cuda") = USE_CUDA);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}