#pragma once

#include "../Layer.h"

#include <iostream>
#include <random>

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
		CLProgram::queueKernel(this->kernelKeys["forward_pass"], this->numWeightedValGlobal, { 1 });
		this->nextLayer->forwardPass();
	}

	std::vector<std::vector<Datatype>> returnNetworkValues() override {
		std::vector<std::vector<Datatype>> returnValues;
		returnValues.push_back(this->neuronValues);
		std::vector<std::vector<Datatype>> nextLayerValues = this->nextLayer->returnNetworkValues();
		returnValues.insert(returnValues.end(), nextLayerValues.begin(), nextLayerValues.end());
		return returnValues;
	}

	void assignNextLayers(Layer<Datatype>* nextL) override {
		this->nextLayer = nextL;
	}

	std::vector<Datatype> predict(std::vector<Datatype> inputs, Datatype bias = 1.0f) {
		this->setNeuronValues(inputs);
		inputs.push_back(bias);
		// Copy the value of input1 and input2 to the buffer
		CLProgram::writeBuffer<float>("inOutValues", 0, inputs);
		forwardPass();
		std::vector<std::vector<Datatype>> networkValues = returnNetworkValues();
		return networkValues[networkValues.size() - 1];
	}

	void learn(std::vector<Datatype> inputs, std::vector<Datatype> correctOutputs, bool printEpoch, Datatype bias = 1.0f) {
		std::vector<Datatype> predictedOutput = predict(inputs, bias);

		CLProgram::queueKernel("reset_depth", { 1 }, { 1 });
		CLProgram::writeBuffer<float>("correctOutput", 0, correctOutputs);

		CLProgram::queueKernel(this->kernelKeys["learn"], this->numWeightedValGlobal, { 1 });
		this->nextLayer->learn();

		if (printEpoch) {
			std::cout << "    Testing (" << inputs[0] << ", " << inputs[1] << "): { Output: " << predictedOutput[0] << ", Expected: " << correctOutputs[0] << ", Error: " << predictedOutput[0] - correctOutputs[0] << " }" << std::endl;
		}
	}

	unsigned int getOffset() override {
		return this->numWeightedValues();
	}
};