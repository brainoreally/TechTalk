
__kernel void reset_depth(
    __global unsigned int* layerDepth
) {
    layerDepth[0] = 0;
}

__kernel void dot_product_forward_pass(
    __global unsigned int* layerDepth,
    __global unsigned int* valueOffsets,
    __global float* layerValues,
    __global float* inputValues,
    __global float* weightValues,
    __global float* networkValues
)
{
    const int gid = get_global_id(0);
    networkValues[valueOffsets[layerDepth[0]] + gid] = inputValues[gid];
    layerValues[gid] = inputValues[gid] * weightValues[valueOffsets[layerDepth[0]] + gid];
}

__kernel void advance_layer(
    __global unsigned int* layerDepth
) {
    layerDepth[0]++;
}

__kernel void sigmoid_activation(
    __global unsigned int* layerDepth,
    __global unsigned int* valueOffsets,
    __global float* layerValues,
    __global float* outputValues
)
{
    const int gid = get_global_id(0);

    float unNormalizedOutput = 0.0f;

    for(int i = 0; i < valueOffsets[layerDepth[0]]; i++)
        unNormalizedOutput += layerValues[i];

    outputValues[gid] = 1.0f / (1.0f + exp(-unNormalizedOutput));
}

__kernel void perceptron_learn(
    __global unsigned int* layerDepth,
    __global unsigned int* valueOffsets,
    __global float* networkValues,
    __global float* weightValues,
    __global float* outputValues,
    __global float* correctOutput
)
{
    const int gid = get_global_id(0);

    float err = correctOutput[0] - outputValues[0];
    float learningRate = 1.0f;
    weightValues[valueOffsets[layerDepth[0]] + gid] += (err * networkValues[valueOffsets[layerDepth[0]] + gid] * learningRate);
}

__kernel void add_outputs_to_network_values(
    __global unsigned int* layerDepth,
    __global unsigned int* valueOffsets,
    __global float* networkValues,
    __global float* outputValues
)
{
    const int gid = get_global_id(0);
    networkValues[valueOffsets[layerDepth[0]] + gid] = outputValues[gid];
}
