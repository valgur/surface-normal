# surface-normal
This is a rewrite of [JiaxiongQ/surface-normal](https://github.com/JiaxiongQ/surface-normal) –
a data preprocessing tool for [JiaxiongQ/DeepLiDAR](https://github.com/JiaxiongQ/DeepLiDAR).

The `calplanenormal()` from the original code is provided as a setup.py-installable Python
library with a single function: `surface_normal.normals_from_depth`.

Also uses CUDA for processing, if available, which cuts down the processing time for the full KITTI depth dataset from hours to just minutes.

## Setup

Build and install the library. Requires CMake 3.11+, OpenCV 3+ and Eigen3. Both Python 2 and 3 are supported.

```bash
sudo apt-get install cmake libopencv-dev libeigen3-dev

python setup.py install
```

For CUDA support, the following environment variables might need to be set with appropriate values:

```bash
export CUDACXX=/usr/local/cuda-10.2/bin/nvcc
# if the default compiler is not yet supported by CUDA
export CUDAHOSTCXX=/usr/bin/g++-8
```

## Usage

Takes a depth image (such as the ones provided with the [KITTI depth completion dataset](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion))
as input and outputs estimated normals as an RGB image.
The scale of the depth values does not matter.

```python
from surface_normal import normals_from_depth

# Camera intrinsics
f  = 721.5377
cx = 609.5593
cy = 172.8540
normals_from_depth("depth.png", "normals.png",
    intrinsics=(f, cx, cy),
    window_size=15,
    max_rel_depth_diff=0.1
)
```

Alternatively, `numpy.ndarray` input-output is available with `normals_from_depth_numpy()`.
```python
from surface_normal import normals_from_depth_numpy
from skimage.io import imread, imsave

# Camera intrinsics
f  = 721.5377
cx = 609.5593
cy = 172.8540
depth = imread("depth.png")
normals = normals_from_depth_numpy(depth, intrinsics=(f, cx, cy), window_size=15, max_rel_depth_diff=0.1)
imsave("normals.png", normals)
```

### Depth input example 

![depth](depth.png)

### Normals output
![normals](normals.png)

## Citation
If you use the code or method in your work, please cite the following:  
```
@inproceedings{qiu2018deeplidar,
  title={DeepLiDAR: Deep Surface Normal Guided Depth Prediction for Outdoor Scene from Sparse LiDAR Data and Single Color Image},
  author={Qiu, Jiaxiong and Cui, Zhaopeng and Zhang, Yinda and Zhang, Xingdi and Liu, Shuaicheng and Zeng, Bing and Pollefeys, Marc},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
