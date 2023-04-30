#pragma once

#include "../Layer.h"

#include <random>

template<typename Datatype>
class HiddenLayer : public Layer<Datatype>
{
public:
	HiddenLayer<Datatype>() : Layer<Datatype>() {}
	HiddenLayer<Datatype>(LayerParams params, Layer<Datatype>* previousLayer) : Layer<Datatype>(params) {}
	~HiddenLayer() {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<Datatype> dis(0.0f, 1.0f);

		for (int i = 0; i < this->numWeightedValues(); i++) {
			this->weights.push_back(dis(gen));
		}

		CLProgram::writeBuffer(this->bufferKeys["weights"], 0, this->weights);
	}

	void forwardPass() override
	{
		// Queue our forward pass
		CLProgram::queueKernel(this->kernelKeys["forward_pass"]);
		CLProgram::queueKernel(this->kernelKeys["activate"]);

		this->nextLayer->forwardPass();
	}

	void assignNextLayers(Layer<Datatype>* nextL) override {
		previousLayer->assignNextLayers(this);
	}

	std::vector<std::vector<Datatype>> returnNetworkValues() override {
		std::vector<std::vector<Datatype>> returnValues;
		returnValues.push_back(this->neuronValues);
		std::vector<std::vector<Datatype>> nextLayerValues = this->nextLayer->returnNetworkValues();
		returnValues.insert(returnValues.end(), nextLayerValues.begin(), nextLayerValues.end());
		return returnValues;
	}
private:
	Layer<Datatype>* nextLayer;
	Layer<Datatype>* previousLayer;
};