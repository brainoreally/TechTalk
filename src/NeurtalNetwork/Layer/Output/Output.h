#pragma once

#include "../Layer.h"

class OutputLayer : public Layer
{
public:
	OutputLayer();
	OutputLayer(int numNeurons);
	~OutputLayer();

	void forwardPass();
	std::vector<std::vector<float>> returnNetworkValues();
};