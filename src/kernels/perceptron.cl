float dot_prod(
	__global float* weights,
	__global float* neuronValues,
	int inputSize,
	int inputValOffset,
	int weightOffset
)
{
	float sum = 0.0f;
	for (int iter = 0; iter < inputSize; iter++) {
		sum += neuronValues[inputValOffset + iter] * weights[weightOffset + iter];
	}
	return sum;
}

float sigmoid_derivitive(
	float value
)
{
	return value * (1.0f - value);
}

float sigmoid_activation(
	float value
)
{
	return 1.0f / (1.0f + exp(-value));
}

float relu_derivitive(
	float value
)
{
	if (value > 0.0f) {
		return 1.0f;
	}
	else {
		return 0.0f;
	}
}

float relu_activation(
	float value
)
{
	return max(0.0f, value);
}

__kernel void network_output(
	__global unsigned int* networkCounts,
	__global unsigned int* layerSizes,
	__global float* correctOutput,
	__global float* neuronValues,
	__global float* weights,
	__global float* biases
)
{
	if (networkCounts[6] % networkCounts[7] == 0) {
		float loss = 0.0f;
		int outputLayerOff = networkCounts[0] - networkCounts[4];

		for (unsigned int sampleIter = 0; sampleIter < networkCounts[8]; sampleIter++)
			for (unsigned int neuronIter = 0; neuronIter < networkCounts[4]; neuronIter++)
				loss += correctOutput[(sampleIter * networkCounts[4]) + neuronIter] - neuronValues[(sampleIter * networkCounts[0]) + outputLayerOff + neuronIter];
		loss /= (networkCounts[8] * networkCounts[4]);
		
		printf("Epoch: %i - Error: %f\n", networkCounts[6], loss);

		/*
		printf("Network Values:\n");
		for (int sampleIter = 0; sampleIter < 4; sampleIter++) {
			int layerOffset = 0;
			for (int layerIter = 0; layerIter < networkCounts[2]; layerIter++) {
				printf("    { ");
				for (int neuronIter = 0; neuronIter < layerSizes[layerIter]; neuronIter++) {
					printf("%f, ", neuronValues[(sampleIter * networkCounts[0]) + layerOffset + neuronIter]);
				}
				printf("}, ");
				layerOffset += layerSizes[layerIter];
			}
			printf("\n");
		}
		printf("Bias Values:\n");
		int layerOffset = 0;
		for (int layerIter = 0; layerIter < networkCounts[2]; layerIter++) {
			printf("    { ");
			for (int neuronIter = 0; neuronIter < layerSizes[layerIter]; neuronIter++) {
				printf("%f, ", biases[layerOffset + neuronIter]);
			}
			printf("}, ");
			layerOffset += layerSizes[layerIter];
		}
		printf("\n");
		printf("Weight Values:\n");
		layerOffset = 0;
		for (int layerIter = 1; layerIter < networkCounts[2]; layerIter++) {
			printf("    { ");
			for (int weightIter = 0; weightIter < (layerSizes[layerIter] * layerSizes[layerIter - 1]); weightIter++) {
				printf("%f, ", weights[layerOffset + weightIter]);
			}
			printf("}, ");
			layerOffset += layerSizes[layerIter] * layerSizes[layerIter - 1];
		}
		printf("\n");*/
	}
	networkCounts[9] = 0;
	--networkCounts[6];
}

__kernel void batch_output(
	__global unsigned int* networkCounts
)
{
	//printf("    Batch#: %i\n", networkCounts[9] / networkCounts[8]);
	networkCounts[9] += networkCounts[8];
}

__kernel void forward_pass(
	__global unsigned int* networkCounts,
	__global unsigned int* layerSizes,
	__global unsigned int* layerActivations,
	__global float* neuronValues,
	__global float* weights,
	__global float* biases
)
{
	const unsigned int sampleIndex = (get_global_id(0) / get_local_size(0)) + networkCounts[9];
	const unsigned int nSOff = sampleIndex * networkCounts[0];
	const unsigned int neuronID = get_local_id(0);

	unsigned int currentLayer = 1;
	unsigned int layerOffset = layerSizes[0];
	unsigned int weightOffset = 0;

	while (currentLayer < networkCounts[2]) {
		if (neuronID < layerSizes[currentLayer]) {
			int nOff = nSOff + layerOffset + neuronID;
			if (layerActivations[currentLayer - 1] == 0) {
				neuronValues[nOff] = sigmoid_activation(dot_prod(weights, neuronValues, layerSizes[currentLayer - 1], nSOff + layerOffset - layerSizes[currentLayer - 1], weightOffset + (neuronID * layerSizes[currentLayer - 1])) + biases[layerOffset + neuronID]);
			}
			else {
				neuronValues[nOff] = relu_activation(dot_prod(weights, neuronValues, layerSizes[currentLayer - 1], nSOff + layerOffset - layerSizes[currentLayer - 1], weightOffset + (neuronID * layerSizes[currentLayer - 1])) + biases[layerOffset + neuronID]);
			}
			//printf("nOff: %i, wOff: %i, bOff: %i, bias: %f\n", nOff, weightOffset + (neuronID * layerSizes[currentLayer - 1]), layerOffset + neuronID, biases[layerOffset + neuronID]);
		}
		layerOffset += layerSizes[currentLayer];
		weightOffset += layerSizes[currentLayer - 1] * layerSizes[currentLayer];
		++currentLayer;
	}
}

__kernel void backward_pass(
	__global unsigned int* networkCounts,
	__global unsigned int* layerSizes,
	__global unsigned int* layerActivations,
	__global float* learningRate,
	__global float* correctOutput,
	__global float* neuronValues,
	__global float* weights,
	__global float* biases,
	__global float* weightDerivitiveOut
)
{
	const unsigned int sampleIndex = (get_global_id(0) / get_local_size(0)) + networkCounts[9];
	const unsigned int nSOff = sampleIndex * networkCounts[0];
	const unsigned int wSOff = sampleIndex * networkCounts[1];
	const unsigned int neuronID = get_local_id(0);

	unsigned int currentLayer = networkCounts[2] - 1;

	unsigned int layerOffset = layerSizes[0];
	unsigned int weightOffset = 0;

	for (int iter = 1; iter < currentLayer; iter++) {
		layerOffset += layerSizes[iter];
		weightOffset += layerSizes[iter] * layerSizes[iter - 1];
	}

	//output layer
	if (neuronID < layerSizes[currentLayer]) {
		int nOff = nSOff + layerOffset + neuronID;

		float loss = 0.0f;
		for (int iter = 0; iter < networkCounts[4]; iter++)
			loss += correctOutput[(sampleIndex * networkCounts[4]) + iter] - neuronValues[nSOff + layerOffset + iter];
		loss /= networkCounts[4];

		if (layerActivations[currentLayer - 1] == 0) {
			weightDerivitiveOut[nOff] = loss * sigmoid_derivitive(neuronValues[nOff]);
		}
		else {
			weightDerivitiveOut[nOff] = loss * relu_derivitive(neuronValues[nOff]);
		}
	}

	unsigned int oldWeightOff;
	unsigned int oldLayerOff;
	unsigned int layerInputSize;

	while (currentLayer > 1) {
		currentLayer--;

		oldLayerOff = layerOffset;
		layerOffset -= layerSizes[currentLayer];

		layerInputSize = layerSizes[currentLayer - 1];

		if (neuronID < layerSizes[currentLayer]) {
			int nOff = nSOff + layerOffset + neuronID;
			float loss = 0.0f;
			for (int nextLayerIter = 0; nextLayerIter < layerSizes[currentLayer + 1]; nextLayerIter++)
			{
				int wOff = weightOffset + neuronID + (nextLayerIter * layerSizes[currentLayer]);
				loss += weightDerivitiveOut[nSOff + oldLayerOff + nextLayerIter] * weights[wOff];
			}
			if (layerActivations[currentLayer - 1] == 0) {
				weightDerivitiveOut[nOff] = loss * sigmoid_derivitive(neuronValues[nOff]);
			}
			else {
				weightDerivitiveOut[nOff] = loss * relu_derivitive(neuronValues[nOff]);
			}
		}

		oldWeightOff = weightOffset;
		weightOffset -= layerSizes[currentLayer] * layerSizes[currentLayer - 1];
	}
}

__kernel void train_biases(
	__global unsigned int* networkCounts,
	__global unsigned int* layerSizes,
	__global float* learningRate,
	__global float* neuronValues,
	__global float* biases,
	__global float* weightDerivitiveOut
)
{
	const unsigned int biasID = get_global_id(0);
	if (biasID < networkCounts[0] && biasID >= networkCounts[3]) {
		unsigned int currentLayer = 1;
		unsigned int sourceNeuronOffset = 0;
		unsigned int lastTotal = layerSizes[0];

		for (int iter = 0; iter < networkCounts[2]; iter++) {
			if (biasID >= lastTotal) {
				lastTotal += layerSizes[currentLayer];
				sourceNeuronOffset += layerSizes[currentLayer - 1];
				currentLayer++;
			}
		}

		float avgChange = 0.0f;
		for (int sampleIter = 0; sampleIter < networkCounts[8]; sampleIter++) {
			int sampleOff = networkCounts[0] * sampleIter;
			for (int neuronIter = 0; neuronIter < layerSizes[currentLayer - 1]; neuronIter++) {
				//printf("biasID: %i, index: %i, weightDerivitiveOut: %f,  nindex: %i, nVal: %f\n", biasID, sampleOff + sourceNeuronOffset + neuronIter, weightDerivitiveOut[sampleOff + sourceNeuronOffset + neuronIter], sampleOff + biasID, neuronValues[sampleOff + biasID]);
				avgChange += weightDerivitiveOut[sampleOff + sourceNeuronOffset + neuronIter] * neuronValues[sampleOff + biasID] * learningRate[0];
			}
		}
		//printf("avgChange: %f\n", avgChange);
		biases[biasID] += avgChange / networkCounts[8];
	}
}

__kernel void train_weights(
	__global unsigned int* networkCounts,
	__global unsigned int* layerSizes,
	__global float* learningRate,
	__global float* neuronValues,
	__global float* weights,
	__global float* weightDerivitiveOut
)
{
	const unsigned int weightID = get_global_id(0);
	
	if (weightID < networkCounts[1]) {
		unsigned int currentLayer = 1;
		unsigned int sourceNeuronOffset = 0;
		unsigned int resultNeuronOffset = layerSizes[0];
		unsigned int lastTotal = 0;

		for (int iter = 0; iter < networkCounts[2]; iter++) {
			if (weightID >= lastTotal + (layerSizes[currentLayer] * layerSizes[currentLayer - 1])) {
				lastTotal += layerSizes[currentLayer] * layerSizes[currentLayer - 1];
				resultNeuronOffset += layerSizes[currentLayer];
				sourceNeuronOffset += layerSizes[currentLayer - 1];
				currentLayer++;
			}
		}
		sourceNeuronOffset += (weightID - lastTotal) % layerSizes[currentLayer - 1];
		resultNeuronOffset += (weightID - lastTotal) / layerSizes[currentLayer - 1];

		float avgChange = 0.0f;
		for (int sampleIter = 0; sampleIter < networkCounts[8]; sampleIter++) {
			int sampleOff = networkCounts[0] * sampleIter;
			avgChange += weightDerivitiveOut[sampleOff + resultNeuronOffset] * neuronValues[sampleOff + sourceNeuronOffset] * learningRate[0];
			//printf("sourceNeuronOffset: %i, sampleOff: %i, neuronValues: %f\n", sourceNeuronOffset, sampleOff, neuronValues[sampleOff + sourceNeuronOffset]);
		}

		weights[weightID] += avgChange;
	}
}