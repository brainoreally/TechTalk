#pragma once

#include "../Layer.h"

#include <random>
#include <iostream>

template<typename Datatype>
class OutputLayer : public Layer<Datatype>
{
public:
	OutputLayer<Datatype>() : Layer<Datatype>() {}
	OutputLayer<Datatype>(LayerParams params) 
		: Layer<Datatype>(params) { }
	~OutputLayer() {}

	std::vector<std::vector<Datatype>> returnNetworkValues() override {
		std::vector<std::vector<Datatype>> returnValues;
		this->setNeuronValues(CLProgram::readBuffer<float>("neuronValues", this->previousLayer->getNeuronValueOffset(), this->numNeurons));
		returnValues.push_back(this->neuronValues);
		return returnValues;
	}
};