#pragma once

#include "../CLProgram/CLProgram.h"
struct LayerParams {
	LayerParams() {}
	LayerParams(int numN, std::map<const char*, const char*> kernKey, std::map<const char*, const char*> buffKey) :
		numNeurons(numN), kernelKeys(kernKey), bufferKeys(buffKey) { }
	std::map<const char*, const char*> kernelKeys;
	std::map<const char*, const char*> bufferKeys;
	int numNeurons;
};

class Layer
{
public:
	Layer() {};

	Layer(LayerParams params) {
		numNeurons = params.numNeurons;
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