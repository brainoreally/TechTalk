#pragma once

#include "../Layer.h"

class InputLayer : public Layer
{
public:
	InputLayer();
	InputLayer(Layer* nextLayer, int numNeurons);
	~InputLayer();

	std::vector<float> predict(std::vector<float> inputs);
	void forwardPass();
	std::vector<std::vector<float>> returnNetworkValues();
	void learn(std::vector<float> inputs, std::vector<float> outputs, bool printEpoch);
protected:
	Layer* nextLayer;
};