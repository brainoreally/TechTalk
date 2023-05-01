#pragma once

#include <vector>
#include <map>

struct LayerParams {
    LayerParams() : dimX(0), dimY(0), dimZ(0), kernelKeys({}) {}
    LayerParams(int dataXlength, int dataYlength, int dataZlength, std::map<const char*, const char*> kernKey) :
        dimX(dataXlength), dimY(dataYlength), dimZ(dataZlength), kernelKeys(kernKey) { }

    std::map<const char*, const char*> kernelKeys;
    int dimX;
    int dimY;
    int dimZ;

    int numNeurons() { return dimX * dimY * dimZ; }
};

struct KernelParam {
    const char* key;
    std::vector<const char*> param_buffer_keys;

    KernelParam(const char* k, std::vector<const char*> bufferK)
        : key(k), param_buffer_keys(bufferK)
    {}
};

struct NetworkParams {
    NetworkParams() :
        inputLayerParams(LayerParams()), hiddenLayerParams({}), outputLayerParams(LayerParams()) {}
    NetworkParams(LayerParams inputLayerParam, std::vector<LayerParams> hiddenLayerParam, LayerParams outputLayerParam) :
        inputLayerParams(inputLayerParam), hiddenLayerParams(hiddenLayerParam), outputLayerParams(outputLayerParam) {}

    LayerParams inputLayerParams;
    std::vector<LayerParams> hiddenLayerParams;
    LayerParams outputLayerParams;
    std::vector<unsigned int> valueOffsets;
    std::vector<unsigned int> layerInputSize;

    std::vector<KernelParam> kernel_params;
    const char* kernel_source_path;
    int totalNumOutputs;
    int totalNumWeights;
    int maxValuesInLayer;
    int outputLayersize;
    int numLayersInNetwork;
};
