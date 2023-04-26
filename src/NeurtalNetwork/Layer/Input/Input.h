#pragma once

#include "../Layer.h"

class InputLayer : public Layer
{
public:
	InputLayer();
	InputLayer(Layer* nextLayer, int numNeurons);
	~InputLayer();

	std::vector<float> forwardPass(std::vector<float> inputs);
	std::vector<std::vector<float>> returnNetworkValues();
	void learn(std::vector<float> inputs, std::vector<float> outputs, bool printEpoch);
protected:
	Layer* nextLayer;
	std::vector<float> activate();
};