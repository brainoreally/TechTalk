#pragma once

#include "../CLProgram/CLProgram.h"

struct LayerParams {
	LayerParams() : dimX(0), dimY(0), dimZ(0), kernelKeys({}), bufferKeys({}) {}
	LayerParams(int dataXlength, int dataYlength, int dataZlength, std::map<const char*, const char*> kernKey, std::map<const char*, const char*> buffKey) :
		dimX(dataXlength), dimY(dataYlength), dimZ(dataZlength), kernelKeys(kernKey), bufferKeys(buffKey) { }

	std::map<const char*, const char*> kernelKeys;
	std::map<const char*, const char*> bufferKeys;
	int dimX;
	int dimY;
	int dimZ;

	int numNeurons() { return dimX * dimY * dimZ; }
};

class Layer
{
public:
	Layer() : numNeurons(0), weights({}), neuronValues({}), kernelKeys({}), bufferKeys({}) {};

	Layer(LayerParams params) {
		numNeurons = params.numNeurons();
		kernelKeys = params.kernelKeys;
		bufferKeys = params.bufferKeys;

		for (int i = 0; i < numNeurons; i++) {
			neuronValues.push_back(0.0f);
		}
	}

	virtual std::vector<std::vector<float>> returnNetworkValues() { return { {} }; }
	virtual void forwardPass() { }
	virtual void assignNextLayers(Layer * nextLayer) { }
protected:
	int numNeurons;
	std::vector<float> weights;
	std::vector<float> neuronValues;
	std::map<const char*, const char*> kernelKeys;
	std::map<const char*, const char*> bufferKeys;
};