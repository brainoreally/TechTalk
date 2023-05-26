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
    NetworkParams(const char* kernelSourcePath, unsigned int numInput, int numOutput, unsigned int outputActivation, std::vector<std::pair<int, std::vector<int>>> hiddenLayerParam, int numSample) {
        kernel_source_path = kernelSourcePath;
        numSamples = numSample;
        numInputs = numInput;
        layerSizes = { numInputs };
        layerActivations = { };
        numOutputs = numOutput;

        inputLayerParams = LayerParams(numInputs, 1, 1);

        numNeurons = numInputs;
        numWeights = 0;
        numLayers = 1;

        maxNeuronInFwd = numOutputs;

        int previousLayerSize = numInputs;
        for (std::pair<int, std::vector<int>> hiddenParams : hiddenLayerParam)
        {
            for (int hiddenL : hiddenParams.second) {
                LayerParams hiddenP = LayerParams(hiddenL, 1, 1);
                if (hiddenL > maxNeuronInFwd)
                    maxNeuronInFwd = hiddenL;

                layerSizes.push_back(hiddenL);
                hiddenLayerParams.push_back(hiddenP);
                layerActivations.push_back(hiddenParams.first);

                numNeurons += hiddenL;
                numWeights += hiddenL * previousLayerSize;
                ++numLayers;
                previousLayerSize = hiddenL;
            }
        }

        ++numLayers;
        layerSizes.push_back(numOutputs);
        layerActivations.push_back(outputActivation);
        numWeights += numOutputs * previousLayerSize;
        numNeurons += numOutputs;
        outputLayerParams = LayerParams(numOutputs, 1, 1);
    }

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
    unsigned int maxNeuronInFwd = 0;
};
