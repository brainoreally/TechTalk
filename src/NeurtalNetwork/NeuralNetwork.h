#pragma once

#include "Layer/Input/Input.h"
#include "Layer/Output/Output.h"

#include <iostream>
#include <sstream>
#include <fstream>

#include <thread>

struct NetworkParams {
	NetworkParams() :
		inputLayerParams(LayerParams()), outputLayerParams(LayerParams()) {}
	NetworkParams(LayerParams inputLayerParam, LayerParams outputLayerParam) :
		inputLayerParams(inputLayerParam), outputLayerParams(outputLayerParam) {}

	LayerParams inputLayerParams;
	LayerParams outputLayerParams;
};

template<typename Datatype>
class NeuralNetwork {
public:
	NeuralNetwork<Datatype>() { }
	NeuralNetwork<Datatype>(NetworkParams params) : parameters(params) {
		inputLayer = InputLayer<Datatype>(parameters.inputLayerParams);
		Layer<Datatype>* previousLayer = &inputLayer;

		outputLayer = OutputLayer<Datatype>(parameters.outputLayerParams, previousLayer);

		//Have layers pass pointers to the Layer that comes after it: done after creation as they don't exist on creation,
		//   previous Layer pointers are handled on creation as you can pass the last created Layer in.
		inputLayer.assignNextLayers(&outputLayer);

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
	OutputLayer<Datatype> outputLayer;

	uint32_t epoch, cyclesLeft;
	NetworkParams parameters;

	bool earlyEnd;
};