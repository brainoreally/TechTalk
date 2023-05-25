#pragma once

#include "../Layer.h"

template<typename Datatype>
class HiddenLayer : public Layer<Datatype>
{
public:
	HiddenLayer<Datatype>() : Layer<Datatype>() {}
	HiddenLayer<Datatype>(LayerParams params) : 
		Layer<Datatype>(params) 
	{
	}

	~HiddenLayer<Datatype>() {}

	std::vector<std::vector<Datatype>> returnNetworkValues(unsigned int offset) override {
		std::vector<std::vector<Datatype>> returnValues;
		std::vector<Datatype> nVals = CLProgram::readBuffer<float>("neuronValues", offset + this->previousLayer->getNeuronValueOffset(), this->numNeurons);
		this->setNeuronValues(nVals);
		returnValues.push_back(this->neuronValues);
		std::vector<std::vector<Datatype>> nextLayerValues = this->nextLayer->returnNetworkValues(offset);
		returnValues.insert(returnValues.end(), nextLayerValues.begin(), nextLayerValues.end());
		return returnValues;
	}
	
	std::vector<std::vector<Datatype>> returnWeightValues() override {
		std::vector<std::vector<Datatype>> returnValues;
		returnValues.push_back(CLProgram::readBuffer<float>("weights", this->previousLayer->getWeightsOffset(), this->numWeights()));
		std::vector<std::vector<Datatype>> nextLayerValues = this->nextLayer->returnWeightValues();
		returnValues.insert(returnValues.end(), nextLayerValues.begin(), nextLayerValues.end());
		return returnValues;
	}
	/*
	std::vector<std::vector<Datatype>> returnBiasValues() override {
		std::vector<std::vector<Datatype>> returnValues;
		returnValues.push_back(CLProgram::readBuffer<float>("biases", this->previousLayer->getNeuronValueOffset(), this->numNeurons));
		return returnValues.push_back(this->nextLayer->returnBiasValues());
	}*/
};