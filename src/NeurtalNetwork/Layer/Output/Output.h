#pragma once

#include "../Layer.h"

#include <random>
#include <iostream>

template<typename Datatype>
class OutputLayer : public Layer<Datatype>
{
public:
	OutputLayer<Datatype>() : Layer<Datatype>() {}
	OutputLayer<Datatype>(LayerParams params, Layer<Datatype>* prevLayer) 
		: Layer<Datatype>(params) {
		this->previousLayer = prevLayer;
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<Datatype> dis(0.0f, 1.0f);

		for (int i = 0; i < this->numWeightedValues(); i++) {
			this->weights.push_back(dis(gen));
		}

		CLProgram::writeBuffer<float>("weightedValues", 0, this->weights);
	}
	~OutputLayer() {}

	void forwardPass() override
	{
		CLProgram::queueKernel("advance_layer", { 1 }, { 1 });
		CLProgram::queueKernel(this->kernelKeys["activate"], this->numNeuronGlobal, { 1 });
		CLProgram::queueKernel("add_outputs_to_network_values", this->numNeuronGlobal, { 1 });
	}

	std::vector<std::vector<Datatype>> returnNetworkValues() override {
		std::vector<std::vector<Datatype>> returnValues;
		this->setNeuronValues(CLProgram::readBuffer<float>("networkValues", this->getNetworkValueOffset(), this->numNeurons));
		returnValues.push_back(this->neuronValues);
		return returnValues;
	}

	void assignNextLayers(Layer<Datatype>* nextL = nullptr) override {
		this->previousLayer->assignNextLayers(this);
	}

	void learn() override {
		return;
	}
};