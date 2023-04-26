#pragma once

#include "../Layer.h"

class OutputLayer : public Layer
{
public:
	OutputLayer();
	OutputLayer(LayerParams params, Layer* previousLayer);
	~OutputLayer();

	void forwardPass() override;
	std::vector<std::vector<float>> returnNetworkValues() override;
private:
	Layer* previousLayer;
};