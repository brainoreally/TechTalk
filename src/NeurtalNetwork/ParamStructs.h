#pragma once

#include <vector>
#include <map>

struct LayerParams {
    LayerParams() : dimX(0), dimY(0), dimZ(0) {}
    LayerParams(unsigned int dataXlength, unsigned int dataYlength, unsigned int dataZlength) :
        dimX(dataXlength), dimY(dataYlength), dimZ(dataZlength) { }

    unsigned int dimX;
    unsigned int dimY;
    unsigned int dimZ;

    unsigned int numNeurons() { return dimX * dimY * dimZ; }
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

    std::vector<unsigned int> layerSizes;
    std::vector<unsigned int> layerActivations;

    const char* kernel_source_path = "";
    unsigned int numInputs = 0;
    unsigned int numOutputs = 0;
    unsigned int numNeurons = 0;
    unsigned int numWeights = 0;
    unsigned int numSamples = 0;
    unsigned int numLayers = 0;
};
