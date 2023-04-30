#pragma once

#include "../Layer.h"

#include <random>

template<typename Datatype>
class OutputLayer : public Layer<Datatype>
{
public:
	OutputLayer<Datatype>() : Layer<Datatype>() {}
	OutputLayer<Datatype>(LayerParams params, Layer<Datatype>* previousLayer) : Layer<Datatype>(params) {}
	~OutputLayer() {}

	void forwardPass() override
	{
		this->setNeuronValues(CLProgram::readBuffer(this->bufferKeys["output"], 0, this->numNeurons));
	}

	std::vector<std::vector<Datatype>> returnNetworkValues() override {
		std::vector<std::vector<Datatype>> returnValues;
		returnValues.push_back(this->neuronValues);
		return returnValues;
	}
private:
	Layer<Datatype>* previousLayer;
};