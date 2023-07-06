#pragma once

#include "../Layer.h"

#include <random>
#include <iostream>

template<typename Datatype>
class OutputLayer : public Layer<Datatype>
{
public:
	OutputLayer<Datatype>() : Layer<Datatype>() {}
	OutputLayer<Datatype>(LayerParams params) 
		: Layer<Datatype>(params) { }
	~OutputLayer() {}

	std::vector<std::vector<Datatype>> returnNetworkValues(unsigned int offset) override {
		std::vector<std::vector<Datatype>> returnValues;
		this->setNeuronValues(CLProgram::readBuffer<float>("neuronValues", offset + this->previousLayer->getNeuronValueOffset(), this->numNeurons));
		returnValues.push_back(this->neuronValues);
		return returnValues;
	}
	
	std::vector<std::vector<Datatype>> returnWeightValues() override {
		std::vector<std::vector<Datatype>> out = {};
		out.push_back(CLProgram::readBuffer<float>("weights", this->previousLayer->getWeightsOffset(), this->numWeights()));
		return out;
	}

	std::vector<std::vector<Datatype>> returnBiasValues() override {
		std::vector<std::vector<Datatype>> out = {};
		out.push_back(CLProgram::readBuffer<float>("biases", this->previousLayer->getNeuronValueOffset(), this->numNeurons));
		return out;
	}

	void finishLayerSetup() override {

		for (int i = 0; i < this->numWeights(); i++) {
			this->weights.push_back(0.0f);
		}

		for (int i = 0; i < this->numNeuronValues(); i++) {
			this->biases.push_back(0.0f);
		}

		CLProgram::writeBuffer<Datatype>("weights", this->previousLayer->getWeightsOffset() * sizeof(Datatype), this->weights);
		CLProgram::writeBuffer<Datatype>("biases", this->previousLayer->getNeuronValueOffset() * sizeof(Datatype), this->biases);
		this->numWeightedValGlobal = this->numWeights();
		this->previousLayer->finishLayerSetup();
	}
};