#pragma once

#include "Layer/Input/Input.h"
#include "Layer/Hidden/Hidden.h"
#include "Layer/Output/Output.h"

#include <iostream>
#include <sstream>
#include <fstream>

#include <thread>

template<typename Datatype>
class NeuralNetwork {
public:
	NeuralNetwork<Datatype>() { }
	NeuralNetwork<Datatype>(NetworkParams params) : parameters(params) {
		CLProgram::initCL(parameters.kernel_source_path);
		CLProgram::setupNetworkOpenCL(&parameters);
		CLProgram::writeBuffer<unsigned int>("valueOffsets", 0, parameters.valueOffsets);
		CLProgram::writeBuffer<unsigned int>("layerInputSize", 0, parameters.layerInputSize);

		inputLayer = InputLayer<Datatype>(parameters.inputLayerParams);

		int valueOffset = 3;

		for (LayerParams hiddenLayerParameters : parameters.hiddenLayerParams) {
			hiddenLayers.push_back(HiddenLayer<Datatype>(hiddenLayerParameters, valueOffset));
			valueOffset += 5;
		}
		outputLayer = OutputLayer<Datatype>(parameters.outputLayerParams);

		if (hiddenLayers.size() > 0) {
			inputLayer.assignNextLayer(&hiddenLayers[0]);
			int lastHiddenIndex = hiddenLayers.size() - 1;

			for(int i = 0; i < lastHiddenIndex; i++)
				hiddenLayers[i].assignNextLayer(&hiddenLayers[i + 1]);

			hiddenLayers[lastHiddenIndex].assignNextLayer(&outputLayer);

			outputLayer.assignPreviousLayer(&hiddenLayers[lastHiddenIndex]);

			for(int i = lastHiddenIndex; i > 0; i--)
				hiddenLayers[i].assignPreviousLayer(&hiddenLayers[i-1]);

			hiddenLayers[0].assignPreviousLayer(&inputLayer);
		}
		else {
			inputLayer.assignNextLayer(&outputLayer);
			outputLayer.assignPreviousLayer(&inputLayer);
		}

		outputLayer.finishLayerSetup();

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

			for (int i = 0; i < trainingData.first.size(); i++) {
				int test = 0;
				int otherTest = 0;
				inputLayer.predict(trainingData.first[i], false);
				CLProgram::writeBuffer<float>("correctOutput", 0, { trainingData.second[i] });
				outputLayer.train();
				if (printEpoch) {
					std::cout << "    Testing (" << trainingData.first[i][0] << ", " << trainingData.first[i][1] << "): { Output: " << outputLayer.returnNetworkValues()[0][0] << ", Expected: " << trainingData.second[i] << " }" << std::endl;
				}
			}
		}
		// If we fail to set this the cleanup code for this class can hang.
		training = false;
	}

	void train(std::pair<std::vector<std::vector<Datatype>>, std::vector<Datatype>>  trainingData, uint32_t cycles, uint32_t epoc) {
		cyclesLeft = cycles;
		epoch = epoc;

		if (!training && cyclesLeft > 0) {
			training = true;
			std::thread training_thread(&NeuralNetwork<Datatype>::learn, this, trainingData);
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