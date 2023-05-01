#pragma once

#include "../CLProgram/CLProgram.h"

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

template<typename Datatype>
class Layer
{
public:
	Layer<Datatype>() : 
		numNeurons(0), weights({}), neuronValues({}), kernelKeys({}) {};

	Layer<Datatype>(LayerParams params) {
		numNeurons = params.numNeurons();
		kernelKeys = params.kernelKeys;

		for (int i = 0; i < numNeurons; i++) {
			neuronValues.push_back(0.0f);
		}
	}

	virtual std::vector<std::vector<Datatype>> returnNetworkValues() { return { {} }; }

	virtual void forwardPass() { }
	virtual void assignNextLayers(Layer * nextLayer) { }

	void setNeuronValues(std::vector<Datatype> neuronVals) {
		neuronValues = neuronVals;
	}
protected:
	int numWeightedValues() {
		//For now we want to weigh our inputs, plus our bias
		//So return num of neurons (inputs) and add 1 space for the bias
		//This will make sure buffer sizes are allocated correctly
		return this->numNeurons + 1;
	}
	int numNeurons;
	std::vector<Datatype> neuronValues;
	std::vector<Datatype> weights;
	std::map<const char*, const char*> kernelKeys;
};