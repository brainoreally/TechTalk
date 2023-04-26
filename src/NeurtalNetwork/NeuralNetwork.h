#pragma once

#include "Layer/Input/Input.h"
#include "Layer/Output/Output.h"

struct NetworkParams {
	NetworkParams() {}
	NetworkParams(int inDimX, int inDimY, int outDimX, int outDimY) :
		inputDimensionsX(inDimX), inputDimensionsY(inDimY), outputDimensionsX(outDimX), outputDimensionsY(outDimY) {}

	int inputDimensionsX;
	int inputDimensionsY;

	int outputDimensionsX;
	int outputDimensionsY;

	LayerParams inputLayerParams;
	LayerParams outputLayerParams;
};

class NeuralNetwork {
public:
	NeuralNetwork();
	NeuralNetwork(NetworkParams params);
	~NeuralNetwork();

	void learn(std::vector<std::vector<std::vector<float>>> trainingData);

	void train(std::vector<std::vector<std::vector<float>>> trainingData, uint32_t cycles, uint32_t epoch);
	void predict(std::vector<float> inputs);
	std::vector<std::vector<float>> returnNetworkValues();

	bool training;
private:
	InputLayer inputLayer;
	OutputLayer outputLayer;

	uint32_t epoch, cyclesLeft;
	NetworkParams parameters;

	bool earlyEnd;
};