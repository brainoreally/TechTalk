#pragma once

#include "../Layer.h"

#include <random>

template<typename Datatype>
class HiddenLayer : public Layer<Datatype>
{
public:
	HiddenLayer<Datatype>() : Layer<Datatype>() {}
	HiddenLayer<Datatype>(LayerParams params, Layer<Datatype>* prevLayer) : 
		Layer<Datatype>(params), previousLayer(prevLayer) {}
	~HiddenLayer() {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<Datatype> dis(0.0f, 1.0f);

		for (int i = 0; i < this->numWeightedValues(); i++) {
			this->weights.push_back(dis(gen));
		}

		CLProgram::writeBuffer<float>("weightValues", 0, this->weights);
	}

	void forwardPass() override
	{
		// Queue our forward pass
		CLProgram::queueKernel("advance_layer", { 1 }, { 1 });
		CLProgram::queueKernel(this->kernelKeys["activate"], this->numNeuronGlobal, { 1 });
		CLProgram::queueKernel(this->kernelKeys["forward_pass"], this->numWeightedValGlobal, { 1 });

		this->nextLayer->forwardPass();
	}

	void assignNextLayers(Layer<Datatype>* nextL) override {
		previousLayer->assignNextLayers(this);
	}

	std::vector<std::vector<Datatype>> returnNetworkValues() override {
		std::vector<std::vector<Datatype>> returnValues;
		this->setNeuronValues(CLProgram::readBuffer<float>("networkValues", 0, this->numNeurons));
		returnValues.push_back(this->neuronValues);
		std::vector<std::vector<Datatype>> nextLayerValues = this->nextLayer->returnNetworkValues();
		returnValues.insert(returnValues.end(), nextLayerValues.begin(), nextLayerValues.end());
		return returnValues;
	}

private:
	Layer<Datatype>* nextLayer;
	Layer<Datatype>* previousLayer;
};