#pragma once

#include "../Layer.h"

template<typename Datatype>
class HiddenLayer : public Layer<Datatype>
{
public:
	HiddenLayer<Datatype>() : Layer<Datatype>() {}
	HiddenLayer<Datatype>(LayerParams params, int valueOffset) : 
		Layer<Datatype>(params) 
	{
		std::vector<Datatype> biasIn = { 1.0f };
		CLProgram::writeBuffer<Datatype>("networkValues", valueOffset + this->numNeurons, biasIn);
	}

	~HiddenLayer<Datatype>() {}

	void forwardPass() override
	{
		// Queue our forward pass
		CLProgram::queueKernel("advance_layer", { 1 }, { 1 });
		CLProgram::queueKernel(this->kernelKeys["activate"], this->numWeightedValGlobal, this->numWeightedValGlobal);
		CLProgram::queueKernel(this->kernelKeys["forward_pass"], this->numWeightedValGlobal, this->numWeightedValGlobal);

		this->nextLayer->forwardPass();
	}

	void train() override
	{
		CLProgram::queueKernel("decrease_layer", { 1 }, { 1 });
		CLProgram::queueKernel(this->kernelKeys["train"], this->numNeuronGlobal, this->numNeuronGlobal);
		this->previousLayer->train();
	}

	void learn() override {
		CLProgram::queueKernel("advance_layer", { 1 }, { 1 });
		CLProgram::queueKernel("set_layer_error", this->numWeightedValGlobal, this->numWeightedValGlobal);
		CLProgram::queueKernel(this->kernelKeys["learn"], this->numWeightedValGlobal, this->numWeightedValGlobal);
		this->nextLayer->learn();
	}

	std::vector<std::vector<Datatype>> returnNetworkValues() override {
		std::vector<std::vector<Datatype>> returnValues;
		std::vector<Datatype> nVals = CLProgram::readBuffer<float>("networkValues", this->previousLayer->getNeuronValueOffset(), this->numNeurons);
		this->setNeuronValues(nVals);
		returnValues.push_back(this->neuronValues);
		std::vector<std::vector<Datatype>> nextLayerValues = this->nextLayer->returnNetworkValues();
		returnValues.insert(returnValues.end(), nextLayerValues.begin(), nextLayerValues.end());
		return returnValues;
	}
};