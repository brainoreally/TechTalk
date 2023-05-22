#pragma once

#include "../Layer.h"

#include <iostream>

template<typename Datatype>
class InputLayer : public Layer<Datatype>
{
public:
	InputLayer<Datatype>() : Layer<Datatype>() {}
	InputLayer<Datatype>(LayerParams params) : Layer<Datatype>(params) {
	}
	~InputLayer() {}

	std::vector<std::vector<Datatype>> returnNetworkValues(unsigned int offset) override {
		std::vector<std::vector<Datatype>> returnValues;
		std::vector<Datatype> nVals = CLProgram::readBuffer<float>("neuronValues", offset, this->numNeurons);
		this->setNeuronValues(nVals);
		returnValues.push_back(this->neuronValues);
		std::vector<std::vector<Datatype>> nextLayerValues = this->nextLayer->returnNetworkValues(offset);
		returnValues.insert(returnValues.end(), nextLayerValues.begin(), nextLayerValues.end());
		return returnValues;
	}

	unsigned int getNeuronValueOffset() override {
		return this->numNeuronValues();
	}

	unsigned int getWeightsOffset() override {
		return 0;
	}

	int numWeights() override {
		return 0;
	}

	void finishLayerSetup() override {

	}
};