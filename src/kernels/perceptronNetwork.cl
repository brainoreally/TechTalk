
__kernel void perceptron_learn(
    __global int* outputLayerValueIndex,
    __global float* networkValues,
    __global float* weightValues,
    __global float* outputValues,
    __global float* correctOutput
)
{
    const int gid = get_global_id(0);

    float err = correctOutput[0] - outputValues[outputLayerValueIndex];
    float learningRate = 1.0f;

    weightValues[gid] += (err * networkValues[gid] * learningRate);
}

__kernel void sigmoid_activation(
    __global, unsigned int* layerDepth,
    __global, unsigned int* layerValueCounts,
    __global, unsigned int* valueOffset,
    __global float* networkValues,
    __global float* weightValues,
    __global float* layerValues,
    __global float* outputValues
)
{
    const int gid = get_global_id(0);

    float unNormalizedOutput = 0.0f;

    for(int i = 0; i < layerValueCounts[layerDepth]; i++)
        unNormalizedOutput += layerValues[i];

    valueOffset[gid] += layerValueCounts[layerDepth];
    ++layerDepth;
    outputValues[gid] = 1.0f / (1.0f + exp(-unNormalizedOutput));
}

__kernel void dot_product_forward_pass(
    __global, unsigned int* valueOffset,
    __global float* layerValues,
    __global float* weightValues,
    __global float* networkValues
)
{
    const int gid = get_global_id(0);
    networkValues[valueOffset[0] + gid] = layerValues[gid];
    layerValues[gid] = layerValues[gid] * weightValues[valueOffset[0] + gid];
}