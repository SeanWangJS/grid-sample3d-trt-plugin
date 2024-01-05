import ctypes

import torch
import torch.nn.functional as F
from cuda import cudart
import tensorrt as trt
import numpy as np

class OutputAllocator(trt.IOutputAllocator):
    def __init__(self):
        trt.IOutputAllocator.__init__(self)
        self.buffers = {}
        self.shapes = {}

    def reallocate_output(self, tensor_name, memory, size, alignment):
        ptr = cudart.cudaMalloc(size)[1]
        self.buffers[tensor_name] = ptr
        return ptr
    
    def notify_shape(self, tensor_name, shape):
        self.shapes[tensor_name] = tuple(shape)

def load_plugin(logger: trt.Logger):
    success = ctypes.CDLL("build/libgrid_sample_3d_plugin.so", mode = ctypes.RTLD_GLOBAL)
    if not success:
        print("load grid_sample_3d plugin error")
        raise Exception()

    trt.init_libnvinfer_plugins(logger, "")

    registry = trt.get_plugin_registry()
    plugin_creator = registry.get_plugin_creator("GridSample3D", "1", "")

    pf_interpolation_mode = trt.PluginField("interpolation_mode", np.array([0], np.int32), trt.PluginFieldType.INT32)
    pf_padding_mode = trt.PluginField("padding_mode", np.array([0], np.int32), trt.PluginFieldType.INT32)
    pf_align_corners = trt.PluginField("align_corners", np.array([0], np.int32), trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([pf_interpolation_mode, pf_padding_mode, pf_align_corners])
    plugin = plugin_creator.create_plugin("grid_sample_3d", pfc)

    return plugin    

def make_network_and_engine(logger: trt.Logger, 
                            plugin: trt.IPluginV2,
                            input_shape: tuple, 
                            grid_shape: tuple,
                            precision = "float32"):
    
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config  = builder.create_builder_config()
    runtime = trt.Runtime(logger)

    if precision == "float32":
        input_layer = network.add_input(name="input", dtype=trt.float32, shape=input_shape)
        grid_layer = network.add_input(name="grid", dtype=trt.float32, shape=grid_shape)
    elif precision == "float16":
        input_layer = network.add_input(name="input", dtype=trt.float16, shape=input_shape)
        grid_layer = network.add_input(name="grid", dtype=trt.float16, shape=grid_shape)
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        raise Exception("Unsupported: {}".format(precision))

    grid_sample_layer = network.add_plugin_v2(inputs=[input_layer, grid_layer], plugin=plugin)
    print(type(grid_sample_layer))
    print(type(grid_sample_layer.get_output(0)))
    network.mark_output(grid_sample_layer.get_output(0))

    engine_string = builder.build_serialized_network(network, config)
    engine = runtime.deserialize_cuda_engine(engine_string)

    return engine

def inference(engine, context, inputs: dict):

    ## setup input
    input_buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        
        array = inputs[name]
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        array = array.astype(dtype)
        array = np.ascontiguousarray(array)

        err, ptr = cudart.cudaMalloc(array.nbytes)
        if err > 0:
            raise Exception("cudaMalloc failed, error code: {}".format(err))
        input_buffers[name] = ptr
        cudart.cudaMemcpy(ptr, array.ctypes.data, array.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        context.set_input_shape(name, array.shape)
        context.set_tensor_address(name, ptr)

    ## setup output
    output_allocator = OutputAllocator()
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue

        context.set_output_allocator(name, output_allocator)
    
    ## execute
    context.execute_async_v3(0)

    ## fetch output
    output = {}
    for name in output_allocator.buffers.keys():
        ptr = output_allocator.buffers[name]
        shape = output_allocator.shapes[name]
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        nbytes = np.prod(shape) * dtype.itemsize
        
        output_buffer = np.empty(shape, dtype = dtype)
        cudart.cudaMemcpy(output_buffer.ctypes.data, ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        output[name] = output_buffer
    
    ## free input buffers
    for name in input_buffers.keys():
        ptr = input_buffers[name]
        cudart.cudaFree(ptr)
    
    ## free output buffers
    for name in output_allocator.buffers.keys():
        ptr = output_allocator.buffers[name]
        cudart.cudaFree(ptr)

    return output


if __name__ == "__main__":
    logger = trt.Logger(trt.Logger.VERBOSE)
    plugin = load_plugin(logger)
    
    input_shape = (1, 32, 16, 64, 64)
    grid_shape = (1, 16, 64, 64, 3)
    output_shape = (1, 32, 16, 64, 64)
    
    input_tensor = torch.randn(*input_shape, dtype=torch.float32)
    input = input_tensor.numpy()
    
    grid_tensor = torch.randn(*grid_shape, dtype=torch.float32)
    grid = grid_tensor.numpy()

    output_ref = F.grid_sample(input_tensor, grid_tensor).numpy()

    inputs = {"input": input, "grid": grid}
    
    engine = make_network_and_engine(logger, plugin, input_shape, grid_shape, "float16")
    context = engine.create_execution_context()

    output = inference(engine, context, inputs)
    output = output["(Unnamed Layer* 0) [PluginV2DynamicExt]_output_0"]
    # print(output_ref)
    diff = (output - output_ref)
    max_index = np.unravel_index(diff.argmax(), diff.shape)
    min_index = np.unravel_index(diff.argmin(), diff.shape)
    print("max diff: {}%".format(diff.max() / output_ref[max_index] * 100))
    print("min diff: {}%".format(diff.min() / output_ref[min_index] * 100))
    # print(output.keys())
    # print(output.keys())
    # print(output["(Unnamed Layer* 0) [PluginV2DynamicExt]_output_0"].shape)