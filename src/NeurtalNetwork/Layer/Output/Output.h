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

	void forwardPass() override
	{
		CLProgram::queueKernel("advance_layer", { 1 }, { 1 });
		CLProgram::queueKernel(this->kernelKeys["activate"], this->numNeuronGlobal, this->numNeuronGlobal);
		CLProgram::queueKernel("add_outputs_to_network_values", this->numNeuronGlobal, this->numNeuronGlobal);
	}

	void train() override
	{
		CLProgram::queueKernel(this->kernelKeys["train"], this->numNeuronGlobal, this->numNeuronGlobal);
		this->previousLayer->train();
	}

	std::vector<std::vector<Datatype>> returnNetworkValues() override {
		std::vector<std::vector<Datatype>> returnValues;
		this->setNeuronValues(CLProgram::readBuffer<float>("networkValues", this->previousLayer->getNeuronValueOffset(), this->numNeurons));
		returnValues.push_back(this->neuronValues);
		return returnValues;
	}

	void learn() override {
		return;
	}
};