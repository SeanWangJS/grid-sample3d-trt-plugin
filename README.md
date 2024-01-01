## TensorRT plugin for 3D GridSample Operator

At current stage, TensorRT(up to version 8.6.1) does not support 3D GridSample operator.

This plugin is a custom implementation of the 3D GridSample operator for TensorRT. It is inspired by the [GridSample operator](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html) from PyTorch, and the code structure is inspired by project [onnxparser-trt-plugin-sample](https://github.com/TrojanXu/onnxparser-trt-plugin-sample).

### Installation

1. Let cmake find nvcc
```shell
export PATH=/usr/local/cuda/bin:$PATH
```
2. Build the plugin with the following commands:
```shell
mkdir build && cd build
cmake .. -DTensorRT_ROOT=/usr/local/tensorrt
make
```
 
### Usage 

for python code (only on Linux platform), load the plugin with:

```python
import ctypes
success = ctypes.CDLL("build/libgrid_sample_3d_plugin.so", mode = ctypes.RTLD_GLOBAL)
```

see [test_grid_sample3d.py](./test/test_grid_sample3d_plugin.py) for more details.