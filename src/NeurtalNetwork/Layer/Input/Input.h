#pragma once

#include "../Layer.h"

class InputLayer : public Layer
{
public:
	InputLayer();
	InputLayer(LayerParams params);
	~InputLayer();

	void forwardPass() override;
	std::vector<std::vector<float>> returnNetworkValues() override;
	void assignNextLayers(Layer * nextLayer) override;

	std::vector<float> predict(std::vector<float> inputs, float bias = 1.0f);
	void learn(std::vector<float> inputs, std::vector<float> outputs, bool printEpoch, float bias = 1.0f);
protected:
	Layer* nextLayer;
};