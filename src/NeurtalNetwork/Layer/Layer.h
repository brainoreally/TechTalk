#pragma once

#include "../CLProgram/CLProgram.h"

class Layer
{
public:
	Layer() {};

	Layer(int numNeuron) {
		numNeurons = numNeuron;

		for (int i = 0; i < numNeurons; i++) {
			neuronValues.push_back(0.0f);
		}
	}

	virtual std::vector<std::vector<float>> returnNetworkValues() { return { {} }; }
	virtual std::vector<float> forwardPass(std::vector<float> inputs) { return { 0 }; }
protected:
	virtual std::vector<float> activate() { return { 0 }; }

	int numNeurons;
	std::vector<float> weights;
	std::vector<float> neuronValues;
};