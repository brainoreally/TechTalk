#pragma once

#include "Layer/Input/Input.h"
#include "Layer/Output/Output.h"

struct NetworkParams {
	NetworkParams() :
		inputLayerParams(LayerParams()), outputLayerParams(LayerParams()) {}
	NetworkParams(LayerParams inputLayerParam, LayerParams outputLayerParam) :
		inputLayerParams(inputLayerParam), outputLayerParams(outputLayerParam) {}

	LayerParams inputLayerParams;
	LayerParams outputLayerParams;
};

class NeuralNetwork {
public:
	NeuralNetwork();
	NeuralNetwork(NetworkParams params);
	~NeuralNetwork();

	void learn(std::pair<std::vector<std::vector<float>>, std::vector<float>>  trainingData);

	void train(std::pair<std::vector<std::vector<float>>, std::vector<float>>  trainingData, uint32_t cycles, uint32_t epoch);
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