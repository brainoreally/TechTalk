#pragma once

#include "../Layer.h"

class OutputLayer : public Layer
{
public:
	OutputLayer();
	OutputLayer(int numNeurons);
	~OutputLayer();

	std::vector<float> forwardPass(std::vector<float> inputs);
	std::vector<std::vector<float>> returnNetworkValues();
};