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
		CLProgram::writeBuffer<unsigned int>("layerSizes", 0, parameters.layerSizes);
		CLProgram::writeBuffer<unsigned int>("layerActivations", 0, parameters.layerActivations);

		numNeurons = parameters.numNeurons;
		numWeights = parameters.numWeights;
		numLayers = parameters.numLayers;

		unsigned int ncOff;
		ncOff = CLProgram::writeBuffer<unsigned int>("networkCounts", 0, numNeurons);
		ncOff = CLProgram::writeBuffer<unsigned int>("networkCounts", ncOff, numWeights);
		ncOff = CLProgram::writeBuffer<unsigned int>("networkCounts", ncOff, numLayers);
		ncOff = CLProgram::writeBuffer<unsigned int>("networkCounts", ncOff, parameters.numInputs);
		ncOff = CLProgram::writeBuffer<unsigned int>("networkCounts", ncOff, parameters.numOutputs);
		CLProgram::writeBuffer<unsigned int>("networkCounts", ncOff, parameters.numSamples);


		inputLayer = InputLayer<Datatype>(parameters.inputLayerParams);

		for (LayerParams hiddenLayerParameters : parameters.hiddenLayerParams) {
			hiddenLayers.push_back(HiddenLayer<Datatype>(hiddenLayerParameters));
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

		int offset = 0;
		numSamples = trainingData.first.size();

		int iter = 0;
		for (std::vector<Datatype> sampleInput : trainingData.first) {
			if (iter < numSamples) {
				iter++;
				CLProgram::writeBuffer<float>("neuronValues", offset, sampleInput);
				offset += numNeurons * sizeof(float);
			}
		}
		CLProgram::writeBuffer<float>("correctOutput", 0, trainingData.second);
		
		std::vector<float> zeroF = {};

		for (int i = 0; i < numNeurons * numSamples; i++)
			zeroF.push_back(0.0f);

		while (cyclesLeft > 0 && !earlyEnd) {
			--cyclesLeft;
			CLProgram::queueKernel("forward_pass", numSamples * numNeurons, numNeurons);
			CLProgram::queueKernel("backward_pass", numSamples * numNeurons, numNeurons);
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
		CLProgram::writeBuffer<float>("neuronValues", 0, inputs);
		CLProgram::queueKernel("forward_pass", numNeurons, numNeurons);
		std::cout << "Output for values (" + std::to_string(inputs[0]) + ", " + std::to_string(inputs[1]) + ") is: " + std::to_string(outputLayer.returnNetworkValues(0)[0][0]) << std::endl;
	}

	std::vector<std::vector<Datatype>> returnNetworkValues() {
		unsigned int offset = 0;

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<Datatype> dis(0, numSamples);

		offset = dis(gen);

		return inputLayer.returnNetworkValues(offset * numNeurons);
	}

	bool training;
	uint32_t cyclesLeft;
private:
	InputLayer<Datatype> inputLayer;
	std::vector<HiddenLayer<Datatype>> hiddenLayers;
	OutputLayer<Datatype> outputLayer;

	uint32_t epoch;
	NetworkParams parameters;

	bool earlyEnd;
	unsigned int numLayers;
	unsigned int numNeurons;
	unsigned int numWeights;
	unsigned int numSamples;
};