#pragma once

#include "../CLProgram/CLProgram.h"

#include <random>

template<typename Datatype>
class Layer
{
public:
	Layer<Datatype>() : 
		numNeurons(0), weights({}), neuronValues({}) {};

	Layer<Datatype>(LayerParams params) {
		numNeurons = params.numNeurons();

		numNeuronGlobal = numNeurons;

		for (int i = 0; i < numNeurons; i++) {
			neuronValues.push_back(0.0f);
		}
	}

	virtual std::vector<std::vector<Datatype>> returnNetworkValues() { return { {} }; }

	virtual unsigned int getNeuronValueOffset() {
		int offset = numNeuronValues();
		offset += previousLayer->getNeuronValueOffset();
		return offset;
	}

	virtual unsigned int getWeightsOffset() {
		int offset = numWeights() * sizeof(float);
		offset += previousLayer->getWeightsOffset();
		return offset;
	}

	void setNeuronValues(std::vector<Datatype> neuronVals) {
		neuronValues = neuronVals;
	}

	Layer<Datatype>* getNextLayer() {
		return nextLayer;
	}


	void assignNextLayer(Layer<Datatype>* nextL = nullptr) {
		nextLayer = nextL;
	}

	void assignPreviousLayer(Layer<Datatype>* prevL) {
		previousLayer = prevL;
	}
	
	virtual void finishLayerSetup() {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<Datatype> dis(0.0f, 1.0f);

		for (int i = 0; i < numWeights(); i++) {
			weights.push_back(dis(gen));
		}

		CLProgram::writeBuffer<Datatype>("weights", previousLayer->getWeightsOffset(), weights);
		numWeightedValGlobal = numWeights();
		previousLayer->finishLayerSetup();
	}

protected:
	Layer<Datatype>* previousLayer;
	Layer<Datatype>* nextLayer;

	int numNeuronValues() {
		return numNeurons;
	}

	virtual int numWeights() {
		return numNeuronValues() * previousLayer->numNeuronValues();
	}

	int numNeurons;
	std::vector<Datatype> neuronValues;
	std::vector<Datatype> weights;

	size_t numNeuronGlobal;
	size_t numWeightedValGlobal;
};