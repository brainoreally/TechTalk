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

float clip(float val, float min, float max) {
	if (val <= 0.0f)
		return min;
	else if (val > max)
		return max;
	else
		return val;
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
	__global unsigned int* layerActivations,
	__global float* correctOutput,
	__global float* neuronValues,
	__global float* weights,
	__global float* biases,
	__global float* bestLoss
)
{
	float loss = 0.0f;
	if (networkCounts[6] % networkCounts[7] == 0) {
		float accuracy = 0.0f;

		int outputLayerOff = networkCounts[0] - networkCounts[4];

		for (unsigned int sampleIter = 0; sampleIter < networkCounts[8]; sampleIter++) {
			float correctIter = 0;
			float maxValue = 0.0f;
			float maxIter = 0;
			for (unsigned int neuronIter = 0; neuronIter < networkCounts[4]; neuronIter++) {

				if (layerActivations[networkCounts[2] - 2] == 2) {
					loss += -correctOutput[(sampleIter * networkCounts[4]) + neuronIter] * log(clip(neuronValues[(sampleIter * networkCounts[0]) + outputLayerOff + neuronIter], 1e-7, 1 - 1e-7));
					if (log(clip(neuronValues[(sampleIter * networkCounts[0]) + outputLayerOff + neuronIter], 1e-7, 1 - 1e-7)) > maxValue) {
						maxValue = log(clip(neuronValues[(sampleIter * networkCounts[0]) + outputLayerOff + neuronIter], 1e-7, 1 - 1e-7));
						maxIter = (sampleIter * networkCounts[0]) + outputLayerOff + neuronIter;
					}
					if (correctOutput[(sampleIter * networkCounts[4]) + neuronIter] == 1)
						correctIter = (sampleIter * networkCounts[4]) + neuronIter;
				}
				else {
					float val = correctOutput[(sampleIter * networkCounts[4]) + neuronIter] - neuronValues[(sampleIter * networkCounts[0]) + outputLayerOff + neuronIter];
					loss += val;
					accuracy += val < 0.01 && val > -0.01;
				}
			}
			if (layerActivations[networkCounts[2] - 2] == 2)
				accuracy += maxIter == correctIter;
		}

		if (layerActivations[networkCounts[2] - 2] == 2) {
			accuracy /= networkCounts[8];
			loss /= networkCounts[8];
		}
		else {
			loss /= networkCounts[8] * networkCounts[4];
			accuracy /= networkCounts[8] * networkCounts[4];
		}

		printf("Epoch: %i - Loss: %f - Acc: %f\n", networkCounts[6], loss, accuracy);
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

		networkCounts[9] = 0;
	}
	--networkCounts[6];

	if ((loss * -1) < bestLoss[0] || bestLoss[0] == 0.0f)
		bestLoss[0] = loss * -1;
	/*else {
		printf("Loss optimisation reached at cycle %i; ending early.\n", networkCounts[6]);
		networkCounts[6] = 0;
	}*/
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
			if (layerActivations[currentLayer - 1] == 1) {
				neuronValues[nOff] = relu_activation(dot_prod(weights, neuronValues, layerSizes[currentLayer - 1], nSOff + layerOffset - layerSizes[currentLayer - 1], weightOffset + (neuronID * layerSizes[currentLayer - 1])) + biases[layerOffset + neuronID]);
			}
			else if (layerActivations[currentLayer - 1] == 2) {
				//softmax activation
				float sum = 0.0f;
				float maxVal;
				for (int neuronIter = 0; neuronIter < layerSizes[currentLayer]; neuronIter++) {
					float nVal = dot_prod(weights, neuronValues, layerSizes[currentLayer - 1], nSOff + layerOffset - layerSizes[currentLayer - 1], weightOffset + (neuronIter * layerSizes[currentLayer - 1])) + biases[layerOffset + neuronIter];
					if (maxVal < nVal)
						maxVal = nVal;
				}
				for (int neuronIter = 0; neuronIter < layerSizes[currentLayer]; neuronIter++) {
					float nVal = dot_prod(weights, neuronValues, layerSizes[currentLayer - 1], nSOff + layerOffset - layerSizes[currentLayer - 1], weightOffset + (neuronIter * layerSizes[currentLayer - 1])) + biases[layerOffset + neuronIter];
					sum += exp(nVal - maxVal);
				}
				neuronValues[nOff] = exp((dot_prod(weights, neuronValues, layerSizes[currentLayer - 1], nSOff + layerOffset - layerSizes[currentLayer - 1], weightOffset + (neuronID * layerSizes[currentLayer - 1])) + biases[layerOffset + neuronID]) - maxVal) / sum;
			} else {
				neuronValues[nOff] = sigmoid_activation(dot_prod(weights, neuronValues, layerSizes[currentLayer - 1], nSOff + layerOffset - layerSizes[currentLayer - 1], weightOffset + (neuronID * layerSizes[currentLayer - 1])) + biases[layerOffset + neuronID]);
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

		if (layerActivations[networkCounts[2] - 2] == 2)
		{
			for (unsigned int sampleIter = 0; sampleIter < networkCounts[8]; sampleIter++) {
				for (unsigned int neuronIter = 0; neuronIter < networkCounts[4]; neuronIter++) {
					loss += -correctOutput[(sampleIter * networkCounts[4]) + neuronIter] * log(clip(neuronValues[(sampleIter * networkCounts[0]) + layerOffset + neuronIter], 1e-7, 1 - 1e-7));
				}
			}

			loss /= networkCounts[8];
		}
		else {
			for (unsigned int neuronIter = 0; neuronIter < networkCounts[4]; neuronIter++) {
				loss += correctOutput[(sampleIndex * networkCounts[4]) + neuronIter] - clip(neuronValues[(sampleIndex * networkCounts[0]) + layerOffset + neuronIter], 1e-7, 1 - 1e-7);
			}
			loss /= networkCounts[4];
		}

		if (layerActivations[currentLayer - 1] == 1) {
			weightDerivitiveOut[nOff] = loss * relu_derivitive(neuronValues[nOff]);
		} else if (layerActivations[currentLayer - 1] == 2) {
			//softmax derivitive
			float sum = 0.0f;
			float maxVal = 0.0f;
			for (int neuronIter = 0; neuronIter < layerSizes[currentLayer]; neuronIter++) {
				float nVal = dot_prod(weights, neuronValues, layerSizes[currentLayer - 1], nSOff + layerOffset - layerSizes[currentLayer - 1], weightOffset + (neuronIter * layerSizes[currentLayer - 1])) + biases[layerOffset + neuronIter];
				if (maxVal < nVal)
					maxVal = nVal;
			}

			for (int neuronIter = 0; neuronIter < layerSizes[currentLayer]; neuronIter++) {
				float nVal = dot_prod(weights, neuronValues, layerSizes[currentLayer - 1], nSOff + layerOffset - layerSizes[currentLayer - 1], weightOffset + (neuronIter * layerSizes[currentLayer - 1])) + biases[layerOffset + neuronIter];
				sum += exp(nVal - maxVal);
			}
			
			float val = exp((dot_prod(weights, neuronValues, layerSizes[currentLayer - 1], nSOff + layerOffset - layerSizes[currentLayer - 1], weightOffset + (neuronID * layerSizes[currentLayer - 1])) + biases[layerOffset + neuronID]) - maxVal);
			weightDerivitiveOut[nOff] = loss * ((val * sum - val * val) / (sum * sum));
			//printf("weightD: %f, loss: %f, val: %f, sum: %f\n", weightDerivitiveOut[nOff], loss, val, sum);
		}
		else {
			weightDerivitiveOut[nOff] = loss * sigmoid_derivitive(neuronValues[nOff]);
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
	const unsigned int biasID = get_global_id(0) + networkCounts[3];
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
	biases[biasID] -= avgChange / networkCounts[8];
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
	const unsigned int layerIndex = (get_global_id(0) / get_local_size(0)) + 1;
	const unsigned int neuronIndex = get_local_id(0);


	unsigned int sourceNeuronOffset = 0;
	unsigned int resultNeuronOffset = layerSizes[0];
	unsigned int firstWeightOffset = 0;
	unsigned int currentLayer = 1;
	for (int i = 1; i < layerIndex; i++) {
		sourceNeuronOffset += layerSizes[i - 1];
		resultNeuronOffset += layerSizes[i];
		firstWeightOffset += layerSizes[i] * layerSizes[i - 1];
		currentLayer++;
	}

	if (neuronIndex < layerSizes[currentLayer]) {
		resultNeuronOffset += neuronIndex;
		int firstWID = firstWeightOffset + (neuronIndex * layerSizes[currentLayer - 1]);

		for (unsigned int weightID = firstWID; weightID < firstWID + layerSizes[currentLayer - 1]; weightID++) {

			unsigned int sNO = sourceNeuronOffset + (weightID - firstWeightOffset) % layerSizes[currentLayer - 1];

			float avgChange = 0.0f;
			for (int sampleIter = 0; sampleIter < networkCounts[8]; sampleIter++) {
				int sampleOff = networkCounts[0] * sampleIter;
				avgChange += weightDerivitiveOut[sampleOff + resultNeuronOffset] * neuronValues[sampleOff + sNO] * learningRate[0];
				//printf("sourceNeuronOffset: %i, sampleOff: %i, neuronValues: %f\n", sourceNeuronOffset, sampleOff, neuronValues[sampleOff + sourceNeuronOffset]);
			}
			//printf("weightID: %i, avgChange: %f\n", weightID, avgChange);

			weights[weightID] -= avgChange;
		}
	}
}