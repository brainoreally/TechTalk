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
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<Datatype> dis(0.0f, 1.0f);

		for (int i = 0; i < this->numWeightedValues(); i++) {
			this->weights.push_back(dis(gen));
		}

		CLProgram::writeBuffer(this->bufferKeys["weights"], 0, this->weights);
	}
	~InputLayer() {}

	void forwardPass() override {
		// Queue our forward pass
		CLProgram::queueKernel(this->kernelKeys["forward_pass"]);
		CLProgram::queueKernel(this->kernelKeys["activate"]);

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
		nextLayer = nextL;
	}

	std::vector<Datatype> predict(std::vector<Datatype> inputs, Datatype bias = 1.0f) {
		this->setNeuronValues(inputs);
		inputs.push_back(bias);
		// Copy the value of input1 and input2 to the buffer
		CLProgram::writeBuffer(this->bufferKeys["layer_inputs"], 0, inputs);
		forwardPass();
		std::vector<std::vector<Datatype>> networkValues = returnNetworkValues();
		return networkValues[networkValues.size() - 1];
	}

	void learn(std::vector<Datatype> inputs, std::vector<Datatype> correctOutputs, bool printEpoch, Datatype bias = 1.0f) {
		std::vector<Datatype> predictedOutput = predict(inputs, bias);

		CLProgram::writeBuffer(this->bufferKeys["correctOutput"], 0, correctOutputs);
		CLProgram::queueKernel(this->kernelKeys["learn"]);

		if (printEpoch) {
			std::cout << "    Testing (" << inputs[0] << ", " << inputs[1] << "): { Output: " << predictedOutput[0] << ", Expected: " << correctOutputs[0] << ", Error: " << predictedOutput[0] - correctOutputs[0] << " }" << std::endl;
		}
	}
protected:
	Layer<Datatype>* nextLayer;
};