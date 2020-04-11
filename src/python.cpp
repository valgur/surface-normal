#include <string>
#include <tuple>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "surface_normal/numpy.hpp"

namespace py = pybind11;
using namespace surface_normal;

using CameraIntrinsicsTuple = std::tuple<double, double, double>;

py::array_t<uint8_t> normals_from_depth_wrapper(const py::array &depth,
                                                CameraIntrinsicsTuple intrinsics_tuple,
                                                int window_size, float max_rel_depth_diff,
                                                bool use_cuda) {
  CameraIntrinsics intrinsics{};
  intrinsics.f  = std::get<0>(intrinsics_tuple);
  intrinsics.cx = std::get<1>(intrinsics_tuple);
  intrinsics.cy = std::get<2>(intrinsics_tuple);
  return normals_from_depth(depth, intrinsics, window_size, max_rel_depth_diff, use_cuda);
}

PYBIND11_MODULE(surface_normal, m) {
  m.doc() = "";
  m.def("normals_from_depth", &normals_from_depth_wrapper, py::arg("depth"), py::arg("intrinsics"),
        py::arg("window_size") = 15, py::arg("max_rel_depth_diff") = 0.1,
        py::arg("use_cuda") = USE_CUDA);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}