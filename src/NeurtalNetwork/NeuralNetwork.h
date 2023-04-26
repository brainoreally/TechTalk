#pragma once

#include "Layer/Input/Input.h"
#include "Layer/Output/Output.h"

class NeuralNetwork {
public:
	NeuralNetwork();
	~NeuralNetwork();

	void learn();

	void loop();

	bool earlyEnd;
	bool training;

	void train(uint32_t cycles, uint32_t epoch);
	void predict(std::vector<float> inputs);
	std::vector<std::vector<float>> returnNetworkValues();
private:
	InputLayer inputLayer;
	OutputLayer outputLayer;

	void draw();

	uint32_t epoch, cyclesLeft;
};