#include <string.h>
#include <cassert>
#include <iostream>

#include <cuda_fp16.h>
#include <NvInfer.h>

#include "grid_sample_3d_plugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::GridSample3DPlugin;
using nvinfer1::plugin::GridSample3DPluginCreator;

using half = __half;

// plugin specific constants
namespace
{
static const char* GRID_SAMPLER_PLUGIN_VERSION{"1"};
static const char* GRID_SAMPLER_PLUGIN_NAME{"GridSample3D"}; // creator will concat plugintype and namespace
static const char* GRID_SAMPLER_PLUGIN_NAMESPACE{""};
} // namespace

PluginFieldCollection GridSample3DPluginCreator::mFC{};
std::vector<PluginField> GridSample3DPluginCreator::mPluginAttributes;



template <typename scalar_t>
void writeToBuffer(char*& buffer, const scalar_t& val)
{
    *reinterpret_cast<scalar_t*>(buffer) = val;
    buffer += sizeof(scalar_t);
}

template <typename scalar_t>
scalar_t readFromBuffer(const char*& buffer)
{
    scalar_t val = *reinterpret_cast<const scalar_t*>(buffer);
    buffer += sizeof(scalar_t);
    return val;
}

GridSample3DPlugin::GridSample3DPlugin(const std::string name,
                                        size_t inputChannel,
                                        size_t inputDepth,
                                        size_t inputHeight, 
                                        size_t inputWidth,
                                        size_t gridDepth,
                                        size_t gridHeight,
                                        size_t gridWidth,
                                        bool alignCorners,
                                        GridSample3DInterpolationMode interpolationMode,
                                        GridSample3DPaddingMode paddingMode,
                                        DataType dataType):
    mLayerName(name),
    mInputChannel(inputChannel),
    mInputDepth(inputDepth),
    mInputHeight(inputHeight),
    mInputWidth(inputWidth),
    mGridDepth(gridDepth),
    mGridHeight(gridHeight),
    mGridWidth(gridWidth),
    mAlignCorners(alignCorners),
    mInterpolationMode(interpolationMode),
    mPaddingMode(paddingMode),
    mDataType(dataType) {}                                

GridSample3DPlugin::GridSample3DPlugin(const std::string name,
                                       bool alignCorners,
                                       GridSample3DInterpolationMode interpolationMode,
                                       GridSample3DPaddingMode paddingMode):
    mLayerName(name),
    mAlignCorners(alignCorners),
    mInterpolationMode(interpolationMode),
    mPaddingMode(paddingMode) {}                                       

GridSample3DPlugin::GridSample3DPlugin(const std::string name, 
                                       const void *buffer,
                                       size_t buffer_size) {
    
    const char* data = reinterpret_cast<const char*>(buffer); 
    const char* start = data;
    mInputChannel = readFromBuffer<size_t>(data);
    mInputDepth = readFromBuffer<size_t>(data);
    mInputHeight = readFromBuffer<size_t>(data);
    mInputWidth = readFromBuffer<size_t>(data);
    mGridDepth = readFromBuffer<size_t>(data);
    mGridHeight = readFromBuffer<size_t>(data);
    mGridWidth = readFromBuffer<size_t>(data);
    mAlignCorners = readFromBuffer<bool>(data);
    mInterpolationMode = readFromBuffer<GridSample3DInterpolationMode>(data);
    mPaddingMode = readFromBuffer<GridSample3DPaddingMode>(data);
    mDataType = readFromBuffer<DataType>(data);

    assert(data == start + sizeof(size_t) * 7 + sizeof(bool) + sizeof(GridSample3DInterpolationMode) + sizeof(GridSample3DPaddingMode) + sizeof(GridSample3DDataType));
}

GridSample3DPlugin::~GridSample3DPlugin() {}


/***************** IPluginV2DynamicExt Methods *****************/ 

IPluginV2DynamicExt* GridSample3DPlugin::clone() const noexcept {
    auto plugin = new GridSample3DPlugin(mLayerName, 
                                         mInputChannel, 
                                         mInputDepth, 
                                         mInputHeight, 
                                         mInputWidth, 
                                         mGridDepth, 
                                         mGridHeight, 
                                         mGridWidth, 
                                         mAlignCorners, 
                                         mInterpolationMode, 
                                         mPaddingMode, 
                                         mDataType);
    plugin->setPluginNamespace(mNameSpace.c_str());   
    return plugin;                                  
}

DimsExprs GridSample3DPlugin::getOutputDimensions(int32_t outputIndex, 
                                                  DimsExprs const* inputs, 
                                                  int32_t nbInputs, 
                                                  IExprBuilder& exprBuilder) noexcept {
    assert(inputs[0].nbDims == 5);
    assert(inputs[1].nbDims == 5);

    // N, D_grid, H_grid, W_grid, 3
    DimsExprs gridDim = inputs[1];
    DimsExprs output(inputs[0]);
    output.d[2] = gridDim.d[1];
    output.d[3] = gridDim.d[2];
    output.d[4] = gridDim.d[3];
    return output;

}

bool GridSample3DPlugin::supportsFormatCombination(int32_t pos, 
                                                   PluginTensorDesc const* inOut, 
                                                   int32_t nbInputs, 
                                                   int32_t nbOutputs) noexcept {
    assert(nbInputs == 2 && nbOutputs == 1 && pos < (nbInputs + nbOutputs));
    bool condition = inOut[pos].format == TensorFormat::kLINEAR;

    condition &= inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF;
    condition &= inOut[pos].type == inOut[0].type;
    return condition;    
}

void GridSample3DPlugin::configurePlugin(DynamicPluginTensorDesc const* in, 
                                         int32_t nbInputs,
                                         DynamicPluginTensorDesc const* out, 
                                         int32_t nbOutputs) noexcept {
    assert(nbInputs == 2 && nbOutputs == 1);
    // for 3d grid sample, the input should be 5 dims
    assert(in[0].desc.dims.nbDims == 5);        
    assert(in[1].desc.dims.nbDims == 5);

    mBatch = in[0].desc.dims.d[0];
    mInputChannel = in[0].desc.dims.d[1];
    mInputDepth = in[0].desc.dims.d[2];
    mInputHeight = in[0].desc.dims.d[3];
    mInputWidth = in[0].desc.dims.d[4];
    mGridDepth = in[1].desc.dims.d[1];
    mGridHeight = in[1].desc.dims.d[2];
    mGridWidth = in[1].desc.dims.d[3];
    mDataType = in[0].desc.type;

    assert(mBatch == in[1].desc.dims.d[0]);
    assert(in[1].desc.dims.d[4] == 3);

}

size_t GridSample3DPlugin::getWorkspaceSize(PluginTensorDesc const* inputs, 
                                            int32_t nbInputs, 
                                            PluginTensorDesc const* outputs,
                                            int32_t nbOutputs) const noexcept {
    return 0;
}

int32_t GridSample3DPlugin::enqueue(PluginTensorDesc const* inputDesc, 
                                    PluginTensorDesc const* outputDesc,
                                    void const* const* inputs, 
                                    void* const* outputs, 
                                    void* workspace, 
                                    cudaStream_t stream) noexcept{
    int status = -1;                                
    if(mDataType == DataType::kFLOAT) {
        status = grid_sample_3d_cuda<float>(
            static_cast<const float*>(inputs[0]),
            static_cast<const float*>(inputs[1]),
            mBatch, mInputChannel, mInputDepth, mInputHeight, mInputWidth,
            mGridDepth, mGridHeight, mGridWidth,
            mAlignCorners,
            mInterpolationMode,
            mPaddingMode,
            static_cast<float*>(outputs[0]),
            stream
        );
    } else if(mDataType == DataType::kHALF) {
    // } else {
        status = grid_sample_3d_cuda<half>(
            static_cast<const half*>(inputs[0]),
            static_cast<const half*>(inputs[1]),
            mBatch, mInputChannel, mInputDepth, mInputHeight, mInputWidth,
            mGridDepth, mGridHeight, mGridWidth,
            mAlignCorners,
            mInterpolationMode,
            mPaddingMode,
            static_cast<half*>(outputs[0]),
            stream
        );
    }

    return status;
}
/****************** IPluginV2Ext Methods ***************************/
DataType GridSample3DPlugin::getOutputDataType(int32_t index, 
                                         nvinfer1::DataType const* inputTypes, 
                                         int32_t nbInputs) const noexcept {
    assert(index == 0);
    assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    printf("output datatype: %d\n", inputTypes[0]);
    return inputTypes[0];
}

void GridSample3DPlugin::attachToContext(cudnnContext* cudnnContext, 
                                         cublasContext* cublasContext, 
                                         IGpuAllocator* gpuAllocator) noexcept {}

void GridSample3DPlugin::detachFromContext() noexcept {}

/****************** IPluginV2 Methods ******************/

const char* GridSample3DPlugin::getPluginType() const noexcept {
    return GRID_SAMPLER_PLUGIN_NAME;
}

const char* GridSample3DPlugin::getPluginVersion() const noexcept{
    return GRID_SAMPLER_PLUGIN_VERSION;
}

int32_t GridSample3DPlugin::getNbOutputs() const noexcept{
    return 1;
}

int32_t GridSample3DPlugin::initialize() noexcept{
    return 0;
}

void GridSample3DPlugin::terminate() noexcept{}

size_t GridSample3DPlugin::getSerializationSize() const noexcept{
    return sizeof(size_t) * 7 + sizeof(bool) + sizeof(GridSample3DInterpolationMode) + sizeof(GridSample3DPaddingMode) + sizeof(GridSample3DDataType);
}

void GridSample3DPlugin::serialize(void* buffer) const noexcept{
    char* data = reinterpret_cast<char*>(buffer);
    char* start = data;
    writeToBuffer<size_t>(data, mInputChannel);
    writeToBuffer<size_t>(data, mInputDepth);
    writeToBuffer<size_t>(data, mInputHeight);
    writeToBuffer<size_t>(data, mInputWidth);
    writeToBuffer<size_t>(data, mGridDepth);
    writeToBuffer<size_t>(data, mGridHeight);
    writeToBuffer<size_t>(data, mGridWidth);
    writeToBuffer<bool>(data, mAlignCorners);
    writeToBuffer<GridSample3DInterpolationMode>(data, mInterpolationMode);
    writeToBuffer<GridSample3DPaddingMode>(data, mPaddingMode);
    writeToBuffer<DataType>(data, mDataType);
    assert(data == start + getSerializationSize());
}

void GridSample3DPlugin::destroy() noexcept{
    delete this;
}

void GridSample3DPlugin::setPluginNamespace(const char* pluginNamespace) noexcept{
    mNameSpace = pluginNamespace;
}

const char* GridSample3DPlugin::getPluginNamespace() const noexcept{
    return mNameSpace.c_str();
}

/**************************************************/
/********* GridSample3DPluginCreator **************/
/**************************************************/
GridSample3DPluginCreator::GridSample3DPluginCreator() {
    setPluginNamespace(GRID_SAMPLER_PLUGIN_NAMESPACE);
    mFC.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

GridSample3DPluginCreator::~GridSample3DPluginCreator() {}

const char* GridSample3DPluginCreator::getPluginName() const noexcept{
    return GRID_SAMPLER_PLUGIN_NAME;
}

const char* GridSample3DPluginCreator::getPluginVersion() const noexcept{
    return GRID_SAMPLER_PLUGIN_VERSION;
}

const PluginFieldCollection* GridSample3DPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

IPluginV2* GridSample3DPluginCreator::createPlugin(const char* name, 
                                                   const PluginFieldCollection* fc) noexcept {
    const PluginField* fields = fc -> fields;
    int nbFields = fc -> nbFields;
    int interpolationMode = 0;
    int paddingMode = 0;
    int alignCorners = 0;

    for(int i = 0; i < nbFields; i++) {
        const char* field_name = fields[i].name;
        const void* field_data = fields[i].data;

        if(!strcmp(field_name, "interpolation_mode")) { // equal to "interpolation_mode"
            interpolationMode = *(reinterpret_cast<const int*>(field_data));
        }

        if(!strcmp(field_name, "padding_mode")) {
            paddingMode = *(reinterpret_cast<const int*>(field_data));
        }

        if(!strcmp(field_name, "align_corners")) {
            alignCorners = *(reinterpret_cast<const int*>(field_data));
        }
        
    }   

    std::cout << "paddingMode: " << paddingMode << std::endl;
    std::cout << "interpolationMode: " << interpolationMode << std::endl;

    auto plugin = new GridSample3DPlugin(name, 
                                    alignCorners, 
                                    static_cast<GridSample3DInterpolationMode>(interpolationMode), 
                                    static_cast<GridSample3DPaddingMode>(paddingMode));
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;

}

IPluginV2* GridSample3DPluginCreator::deserializePlugin(const char* name, 
                                                        const void* serialData, 
                                                        size_t serialLength) noexcept {
    auto plugin = new GridSample3DPlugin(name, serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void GridSample3DPluginCreator::setPluginNamespace(const char* libNamespace) noexcept {
    mNamespace = libNamespace;
}

const char* GridSample3DPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

extern "C" TENSORRTAPI IPluginCreatorInterface* const* getCreators(int32_t& nbCreators)
{
    nbCreators = 1;
    static GridSample3DPluginCreator sRoiAlignCreator;
    static IPluginCreatorInterface* const kPLUGIN_CREATOR_LIST[] = {&sRoiAlignCreator};
    return kPLUGIN_CREATOR_LIST;
}

extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder* finder)
{

}

REGISTER_TENSORRT_PLUGIN(GridSample3DPluginCreator);
