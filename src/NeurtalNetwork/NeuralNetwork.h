#pragma once

#include "Layer/Input/Input.h"
#include "Layer/Hidden/Hidden.h"
#include "Layer/Output/Output.h"

#include <iostream>
#include <sstream>
#include <fstream>

#include <thread>

struct NetworkParams {
	NetworkParams() :
		inputLayerParams(LayerParams()), hiddenLayerParams({}), outputLayerParams(LayerParams()) {}
	NetworkParams(LayerParams inputLayerParam, std::vector<LayerParams> hiddenLayerParam, LayerParams outputLayerParam) :
		inputLayerParams(inputLayerParam), hiddenLayerParams(hiddenLayerParam), outputLayerParams(outputLayerParam) {}

	LayerParams inputLayerParams;
	std::vector<LayerParams> hiddenLayerParams;
	LayerParams outputLayerParams;
};

template<typename Datatype>
class NeuralNetwork {
public:
	NeuralNetwork<Datatype>() { }
	NeuralNetwork<Datatype>(NetworkParams params) : parameters(params) {

		std::vector<unsigned int> valueOffsets = { 0, 3 };
		CLProgram::writeBuffer<unsigned int>("valueOffsets", 0, valueOffsets);

		inputLayer = InputLayer<Datatype>(parameters.inputLayerParams);
		Layer<Datatype>* previousLayer = &inputLayer;
		for (LayerParams hiddenLayerParameters : parameters.hiddenLayerParams) {
			HiddenLayer<Datatype> hiddenLayer = HiddenLayer<Datatype>(hiddenLayerParameters, previousLayer);
			previousLayer = &hiddenLayer;
			hiddenLayers.push_back(hiddenLayer);
		}
		outputLayer = OutputLayer<Datatype>(parameters.outputLayerParams, previousLayer);

		outputLayer.assignNextLayers();

		training = false;
		earlyEnd = false;
	}

	~NeuralNetwork() {
		// If training thread is running, you'll error on OpenCL code being called if you run the cleanup first.
		// So this is just a basic loop waiting for the thread to stop
		// CAUTION - it assumes errors don't happen/the code successfully sets a boolean value so this loop can hang.
		if (training)
		{
			earlyEnd = true;
			while (training) {
				std::cout << "waiting on training thread to complete..." << std::endl;
			}
		}
		CLProgram::cleanup();
	}


	void learn(std::pair<std::vector<std::vector<Datatype>>, std::vector<Datatype>>  trainingData) {
		if (epoch < 1)
			epoch = 1;

		while (cyclesLeft > 0 && !earlyEnd) {
			bool printEpoch = (cyclesLeft % epoch) == 0;
			if (printEpoch)
				std::cout << "Remaining steps " << cyclesLeft << ":" << std::endl;

			--cyclesLeft;
			for (int i = 0; i < trainingData.first.size(); i++)
				inputLayer.learn(trainingData.first[i], { trainingData.second[i] }, printEpoch);
		}

		// If we fail to set this the cleanup code for this class can hang.
		training = false;
	}

	void train(std::pair<std::vector<std::vector<Datatype>>, std::vector<Datatype>>  trainingData, uint32_t cycles, uint32_t epoc) {
		cyclesLeft = cycles;
		epoch = epoc;

		if (!training && cyclesLeft > 0) {
			training = true;
			std::thread training_thread(&NeuralNetwork::learn, this, trainingData);
			training_thread.detach();
		}
	}

	void predict(std::vector<Datatype> inputs) {
		std::vector<Datatype> outputP = inputLayer.predict(inputs);
		std::cout << "Output for values (" + std::to_string(inputs[0]) + ", " + std::to_string(inputs[1]) + ") is: " + std::to_string(outputP[0]) << std::endl;
	}

	std::vector<std::vector<Datatype>> returnNetworkValues() {
		return inputLayer.returnNetworkValues();
	}

	bool training;
private:
	InputLayer<Datatype> inputLayer;
	std::vector<HiddenLayer<Datatype>> hiddenLayers;
	OutputLayer<Datatype> outputLayer;

	uint32_t epoch, cyclesLeft;
	NetworkParams parameters;

	bool earlyEnd;
};