#include <string>
#include <vector>

#include <NvInferPlugin.h>

#include <grid_sample_3d.h>

#ifndef GRID_SAMPLE_3D_PLUGIN
#define GRID_SAMPLE_3D_PLUGIN

using namespace nvinfer1::plugin;

namespace nvinfer1 {

namespace plugin {

class GridSample3DPlugin : public IPluginV2DynamicExt{

public:

    GridSample3DPlugin(const std::string name,
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
                       nvinfer1::DataType dataType);

    GridSample3DPlugin(const std::string name,
                       bool alignCorners,
                       GridSample3DInterpolationMode interpolationMode,
                       GridSample3DPaddingMode paddingMode);

    GridSample3DPlugin(const std::string name,
                       const void* buffer, 
                       size_t buffer_size);

    GridSample3DPlugin() = delete;
    ~GridSample3DPlugin() override;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, 
                                            DimsExprs const* inputs, 
                                            int32_t nbInputs, 
                                            IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(int32_t pos, 
                                   PluginTensorDesc const* inOut, 
                                   int32_t nbInputs, 
                                   int32_t nbOutputs) noexcept override;

    void configurePlugin(DynamicPluginTensorDesc const* in, 
                         int32_t nbInputs,
                         DynamicPluginTensorDesc const* out, 
                         int32_t nbOutputs) noexcept override;

    size_t getWorkspaceSize(PluginTensorDesc const* inputs, 
                            int32_t nbInputs, 
                            PluginTensorDesc const* outputs,
                            int32_t nbOutputs) const noexcept override;

    int32_t enqueue(PluginTensorDesc const* inputDesc, 
                    PluginTensorDesc const* outputDesc,
                    void const* const* inputs, 
                    void* const* outputs, 
                    void* workspace, 
                    cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int32_t index, 
                                         nvinfer1::DataType const* inputTypes, 
                                         int32_t nbInputs) const noexcept override;
    void attachToContext(cudnnContext* cudnnContext, 
                         cublasContext* cublasContext, 
                         IGpuAllocator* gpuAllocator) noexcept override;

    void detachFromContext() noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    int32_t getNbOutputs() const noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    void destroy() noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:

    const std::string mLayerName;
    size_t mBatch;
    size_t mInputChannel, mInputDepth, mInputWidth, mInputHeight;
    size_t mGridDepth, mGridWidth, mGridHeight;
    bool mAlignCorners;
    std::string mNameSpace;
    GridSample3DInterpolationMode mInterpolationMode;
    GridSample3DPaddingMode mPaddingMode;
    nvinfer1::DataType mDataType;

};

class GridSample3DPluginCreator : public IPluginCreator {

public:
    GridSample3DPluginCreator();
    ~GridSample3DPluginCreator() override;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2* createPlugin(const char* name, 
                            const PluginFieldCollection* fc) noexcept override;

    IPluginV2* deserializePlugin(const char* name, 
                                 const void* serialData, 
                                 size_t serialLength) noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin

} // namespace nvinfer1

#endif // GRID_SAMPLE_3D_PLUGIN
