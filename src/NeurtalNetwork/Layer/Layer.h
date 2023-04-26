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
	virtual void forwardPass() { }
protected:
	int numNeurons;
	std::vector<float> weights;
	std::vector<float> neuronValues;
};