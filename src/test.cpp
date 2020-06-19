#include "surface_normal/opencv2.hpp"

using namespace surface_normal;

int main() {
  CameraIntrinsics intrinsics;
  intrinsics.f  = 721.5377;
  intrinsics.cx = 609.5593;
  intrinsics.cy = 172.8540;
  normals_from_depth_imgfile("depth.png", "normals.png", intrinsics, 15, 0.1, true);
  return 0;
}
