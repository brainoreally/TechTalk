#pragma once

#include "../Layer.h"

#include <random>
#include <iostream>

template<typename Datatype>
class OutputLayer : public Layer<Datatype>
{
public:
	OutputLayer<Datatype>() : Layer<Datatype>(), previousLayer(nullptr) {}
	OutputLayer<Datatype>(LayerParams params, Layer<Datatype>* prevLayer) 
		: Layer<Datatype>(params), previousLayer(prevLayer) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<Datatype> dis(0.0f, 1.0f);

		for (int i = 0; i < this->numWeightedValues(); i++) {
			this->weights.push_back(dis(gen));
		}

		CLProgram::writeBuffer(this->bufferKeys["weights"], 0, this->weights);
	}
	~OutputLayer() {}

	void forwardPass() override
	{
		// Queue our forward pass
		CLProgram::queueKernel(this->kernelKeys["forward_pass"], { 3 }, { 1 });
		CLProgram::queueKernel(this->kernelKeys["activate"], { 1 }, { 1 });

		this->setNeuronValues(CLProgram::readBuffer(this->bufferKeys["output"], 0, this->numNeurons));
	}

	std::vector<std::vector<Datatype>> returnNetworkValues() override {
		std::vector<std::vector<Datatype>> returnValues;
		returnValues.push_back(this->neuronValues);
		return returnValues;
	}

	void assignNextLayers(Layer<Datatype>* nextL = nullptr) override {
		previousLayer->assignNextLayers(this);
	}
private:
	Layer<Datatype>* previousLayer;
};