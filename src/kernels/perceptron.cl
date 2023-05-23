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
	__global float* correctOutput,
	__global float* neuronValues
)
{

	float loss = 0.0f;
	int outputLayerOff = networkCounts[0] - networkCounts[4];
	for (unsigned int sampleIter = 0; sampleIter < networkCounts[5]; sampleIter++)
		for (unsigned int neuronIter = 0; neuronIter < networkCounts[4]; neuronIter++)
			loss += correctOutput[(sampleIter * networkCounts[4]) + neuronIter] - neuronValues[(sampleIter * networkCounts[0]) + outputLayerOff + neuronIter];
	loss /= (networkCounts[5] * networkCounts[4]);
	
	if (networkCounts[6] % networkCounts[7] == 0)
		printf("Epoch: %i - Error: %f\n", networkCounts[6], loss);
	--networkCounts[6];
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
	const unsigned int sampleIndex = get_global_id(0) / get_local_size(0);
	const unsigned int nSOff = sampleIndex * networkCounts[0];
	const unsigned int neuronID = get_local_id(0);

	unsigned int currentLayer = 1;
	unsigned int layerOffset = layerSizes[0];
	unsigned int weightOffset = 0;

	while (currentLayer < networkCounts[2]) {
		if (neuronID < layerSizes[currentLayer]) {
			int nOff = nSOff + layerOffset + neuronID;
			if (layerActivations[currentLayer] == 0) {
				neuronValues[nOff] = sigmoid_activation(dot_prod(weights, neuronValues, layerSizes[currentLayer - 1], nSOff + layerOffset - layerSizes[currentLayer - 1], weightOffset + (neuronID * layerSizes[currentLayer - 1])) + biases[nOff]);
			}
			else {
				neuronValues[nOff] = relu_activation(dot_prod(weights, neuronValues, layerSizes[currentLayer - 1], nSOff + layerOffset - layerSizes[currentLayer - 1], weightOffset + (neuronID * layerSizes[currentLayer - 1])) + biases[nOff]);
			}
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
	__global float* correctOutput,
	__global float* neuronValues,
	__global float* weights,
	__global float* biases,
	__global float* weightDerivitiveOut
)
{
	const unsigned int sampleIndex = get_global_id(0) / get_local_size(0);
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

		float loss = correctOutput[(sampleIndex * networkCounts[4]) + neuronID] - neuronValues[nOff];

		if (layerActivations[currentLayer] == 0) {
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
			if (layerActivations[currentLayer] == 0) {
				weightDerivitiveOut[nOff] = loss * sigmoid_derivitive(neuronValues[nOff]);
			}
			else {
				weightDerivitiveOut[nOff] = loss * relu_derivitive(neuronValues[nOff]);
			}
		}

		oldWeightOff = weightOffset;
		weightOffset -= layerSizes[currentLayer] * layerSizes[currentLayer - 1];
	}	

	barrier(CLK_GLOBAL_MEM_FENCE);
	
	if (neuronID < networkCounts[0] && neuronID >= layerSizes[0] && sampleIndex == 0) {
		unsigned int currentLayer = 1;
		unsigned int sourceNeuronOffset = 0;
		unsigned int lastTotal = layerSizes[0];

		for (int iter = 0; iter < networkCounts[2]; iter++) {
			if (neuronID >= lastTotal) {
				lastTotal += layerSizes[currentLayer];
				sourceNeuronOffset += layerSizes[currentLayer - 1];
				currentLayer++;
			}
		}

		float learningRate = 0.01;
		float avgChange = 0.0f;
		for (int sampleIter = 0; sampleIter < networkCounts[5]; sampleIter++) {
			int sampleOff = networkCounts[0] * sampleIter;
			for(int neuronIter = 0; neuronIter < layerSizes[currentLayer - 1]; neuronIter++)
				avgChange += weightDerivitiveOut[sampleOff + sourceNeuronOffset + neuronIter] * neuronValues[sampleOff + neuronID] * learningRate;
		}

		biases[neuronID] += avgChange / networkCounts[5];
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

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

		float learningRate = 0.01;
		float avgChange = 0.0f;
		for (int sampleIter = 0; sampleIter < networkCounts[5]; sampleIter++) {
			int sampleOff = networkCounts[0] * sampleIter;
			avgChange += weightDerivitiveOut[sampleOff + resultNeuronOffset] * neuronValues[sampleOff + sourceNeuronOffset] * learningRate;
		}

		weights[weightID] += avgChange;
	}
}