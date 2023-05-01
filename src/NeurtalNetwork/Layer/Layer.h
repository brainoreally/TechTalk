#pragma once

#include "../CLProgram/CLProgram.h"

#include <random>

template<typename Datatype>
class Layer
{
public:
	Layer<Datatype>() : 
		numNeurons(0), weights({}), neuronValues({}), kernelKeys({}) {};

	Layer<Datatype>(LayerParams params) {
		numNeurons = params.numNeurons();
		kernelKeys = params.kernelKeys;

		numNeuronGlobal = numNeurons;

		for (int i = 0; i < numNeurons; i++) {
			neuronValues.push_back(0.0f);
		}
	}

	virtual std::vector<std::vector<Datatype>> returnNetworkValues() { return { {} }; }

	virtual void forwardPass() { }

	virtual unsigned int getNeuronValueOffset() {
		int offset = numNeuronValues();
		offset += previousLayer->getNeuronValueOffset();
		return offset;
	}

	virtual unsigned int getWeightedValueOffset() {
		int offset = numWeightedValues();
		offset += previousLayer->getWeightedValueOffset();
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
	
	virtual void train() { }

	virtual void learn() { }

	virtual void finishLayerSetup() {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<Datatype> dis(0.0f, 1.0f);

		for (int i = 0; i < numWeightedValues(); i++) {
			weights.push_back(dis(gen));
		}

		CLProgram::writeBuffer<Datatype>("weightedValues", previousLayer->getWeightedValueOffset(), weights);
		numWeightedValGlobal = numWeightedValues();
		previousLayer->finishLayerSetup();
	}

protected:
	Layer<Datatype>* previousLayer;
	Layer<Datatype>* nextLayer;

	int numNeuronValues() {
		//Our given neuron count plus our bias
		//So return num of neurons (inputs) and add 1 space for the bias
		//This will make sure buffer sizes are allocated correctly
		return numNeurons + 1;
	}

	virtual int numWeightedValues() {
		//Our given neuron count plus our bias
		//So return num of neurons (inputs) and add 1 space for the bias
		//This will make sure buffer sizes are allocated correctly
		return numNeuronValues() * previousLayer->numNeuronValues();
	}

	int numNeurons;
	std::vector<Datatype> neuronValues;
	std::vector<Datatype> weights;
	std::map<const char*, const char*> kernelKeys;

	size_t numNeuronGlobal;
	size_t numWeightedValGlobal;
};