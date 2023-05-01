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

	void forwardPass() override {
		CLProgram::queueKernel("reset_depth", { 1 }, { 1 });
		CLProgram::queueKernel(this->kernelKeys["forward_pass"], this->numWeightedValGlobal, this->numWeightedValGlobal);
		this->nextLayer->forwardPass();
	}

	std::vector<std::vector<Datatype>> returnNetworkValues() override {
		std::vector<std::vector<Datatype>> returnValues;
		returnValues.push_back(this->neuronValues);
		std::vector<std::vector<Datatype>> nextLayerValues = this->nextLayer->returnNetworkValues();
		returnValues.insert(returnValues.end(), nextLayerValues.begin(), nextLayerValues.end());
		return returnValues;
	}
	
	std::vector<Datatype> predict(std::vector<Datatype> inputs, bool skipPredict = false, Datatype bias = 1.0f) {
		this->setNeuronValues(inputs);
		inputs.push_back(bias);
		// Copy the value of input1 and input2 to the buffer
		CLProgram::writeBuffer<float>("inOutValues", 0, inputs);
		this->forwardPass();

		std::vector<Datatype> ret = {};
		if (!skipPredict) {
			std::vector<std::vector<Datatype>> networkValues = returnNetworkValues();
			ret = networkValues[networkValues.size() - 1];
		}
		return ret;
	}

	void train() override
	{
		CLProgram::queueKernel("decrease_layer", { 1 }, { 1 });
		CLProgram::queueKernel(this->kernelKeys["train"], this->numWeightedValGlobal, this->numWeightedValGlobal);
		this->learn();
	}

	void learn() override {
		CLProgram::queueKernel("set_layer_error", this->numWeightedValGlobal, this->numWeightedValGlobal);
		CLProgram::queueKernel(this->kernelKeys["learn"], this->numWeightedValGlobal, this->numWeightedValGlobal);
		this->nextLayer->learn();
	}

	void learn(std::vector<Datatype> inputs, std::vector<Datatype> correctOutputs, bool printEpoch, Datatype bias = 1.0f) {
		std::vector<Datatype> predictedOutput = predict(inputs, bias);

		CLProgram::queueKernel("reset_depth", { 1 }, { 1 });
		CLProgram::writeBuffer<float>("correctOutput", 0, correctOutputs);

		CLProgram::queueKernel(this->kernelKeys["learn"], this->numWeightedValGlobal, this->numWeightedValGlobal);
		this->nextLayer->learn();

		if (printEpoch) {
			std::cout << "    Testing (" << inputs[0] << ", " << inputs[1] << "): { Output: " << predictedOutput[0] << ", Expected: " << correctOutputs[0] << ", Error: " << predictedOutput[0] - correctOutputs[0] << " }" << std::endl;
		}
	}

	unsigned int getNeuronValueOffset() override {
		return this->numNeuronValues();
	}

	unsigned int getWeightedValueOffset() override {
		return this->numNeuronValues();
	}

	int numWeightedValues() override {
		//Our given neuron count plus our bias
		//So return num of neurons (inputs) and add 1 space for the bias
		//This will make sure buffer sizes are allocated correctly
		return this->numNeuronValues();
	}

	void finishLayerSetup() override {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<Datatype> dis(0.0f, 1.0f);

		for (int i = 0; i < this->numWeightedValues(); i++) {
			this->weights.push_back(dis(gen));
		}

		CLProgram::writeBuffer<Datatype>("weightedValues", 0, this->weights);
		this->numWeightedValGlobal = this->numWeightedValues();
	}
};