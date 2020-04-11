#include <string>

#include <pybind11/pybind11.h>

#include "surface_normal/numpy.hpp"
#include "surface_normal/opencv2.hpp"

namespace py = pybind11;
using namespace surface_normal;

void normals_from_depth_imgfile_wrapper(const std::string &depth_in_path,
                                        const std::string &normals_out_path,
                                        CameraIntrinsicsTuple intrinsics_tuple, int window_size,
                                        float max_rel_depth_diff, bool use_cuda) {
  normals_from_depth_imgfile(depth_in_path, normals_out_path, intrinsics_tuple, window_size,
                             max_rel_depth_diff, use_cuda);
}

py::array_t<uint8_t> normals_from_depth_numpy_wrapper(const py::array &depth,
                                                      CameraIntrinsicsTuple intrinsics_tuple,
                                                      int window_size, float max_rel_depth_diff,
                                                      bool use_cuda) {
  return normals_from_depth(depth, intrinsics_tuple, window_size, max_rel_depth_diff, use_cuda);
}

PYBIND11_MODULE(surface_normal, m) {
  m.doc() = "";

  m.def("normals_from_depth", &normals_from_depth_imgfile_wrapper, py::arg("depth_in_path"),
        py::arg("normals_out_path"), py::arg("intrinsics"), py::arg("window_size") = 15,
        py::arg("max_rel_depth_diff") = 0.1, py::arg("use_cuda") = USE_CUDA);

  m.def("normals_from_depth_numpy", &normals_from_depth_numpy_wrapper, py::arg("depth"),
        py::arg("intrinsics"), py::arg("window_size") = 15, py::arg("max_rel_depth_diff") = 0.1,
        py::arg("use_cuda") = USE_CUDA);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}