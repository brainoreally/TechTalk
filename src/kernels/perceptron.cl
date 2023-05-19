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

__kernel void forward_pass(
	__global unsigned int* networkCounts,
	__global unsigned int* layerSizes,
	__global unsigned int* layerActivations,
	__global float* neuronValues,
	__global float* weights
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
				neuronValues[nOff] = sigmoid_activation(dot_prod(weights, neuronValues, layerSizes[currentLayer - 1], nSOff + layerOffset - layerSizes[currentLayer - 1], weightOffset + (neuronID * layerSizes[currentLayer - 1])));
			}
			else {
				neuronValues[nOff] = relu_activation(dot_prod(weights, neuronValues, layerSizes[currentLayer - 1], nSOff + layerOffset - layerSizes[currentLayer - 1], weightOffset + (neuronID * layerSizes[currentLayer - 1])));
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
	__global float* correctOutput,
	__global float* neuronValues,
	__global float* weights,
	__global float* weightLoss,
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

		weightLoss[nOff] = correctOutput[(sampleIndex * networkCounts[4]) + neuronID] - neuronValues[nOff];
		weightDerivitiveOut[nOff] = weightLoss[nOff] * sigmoid_derivitive(neuronValues[nOff]);
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
			weightLoss[nOff] = loss;
			weightDerivitiveOut[nOff] = loss * sigmoid_derivitive(neuronValues[nOff]);
		}

		oldWeightOff = weightOffset;
		weightOffset -= layerSizes[currentLayer] * layerSizes[currentLayer - 1];
	}	
}

__kernel void learn(
	__global unsigned int* networkCounts,
	__global unsigned int* layerSizes,
	__global unsigned int* layerActivations,
	__global float* neuronValues,
	__global float* weights,
	__global float* weightDerivitiveOut,
	__global float* weightLoss
)
{
	const unsigned int weightID = get_local_id(0);

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

	float learningRate = 0.1;
	float avgChange = 0.0f;
	for (int sampleIter = 0; sampleIter < networkCounts[5]; sampleIter++) {
		int sampleOff = networkCounts[0] * sampleIter;
		avgChange += weightDerivitiveOut[sampleOff + resultNeuronOffset] * neuronValues[sampleOff + sourceNeuronOffset] * learningRate;
	}

	weights[weightID] += avgChange;
}