
__kernel void perceptron_learn(__global float* inputs, __global float* weights, __global float* output, __global float* correctOutput)
{
    const int gid = get_global_id(0);

    float modVal = 0.0f;
    if (gid == 2) {
        float bias = 1.0f;
        modVal = bias;
    }
    else {
        modVal = inputs[gid];
    }

    float err = correctOutput[0] - output[0];
    float learningRate = 1.0f;
    weights[gid] += (err * modVal * learningRate);
}

__kernel void sigmoid_activation(__global float* inputs, __global float* weights, __global float* output)
{
    float bias = 1.0f;
    float unNormalizedOutput = (inputs[0]) + (inputs[1]) + (bias * weights[2]);
    output[0] = 1.0f / (1.0f + exp(-unNormalizedOutput));
}

__kernel void dot_product_forward_pass(__global float* inputs, __global float* weights)
{
    const int gid = get_global_id(0);
    inputs[gid] = inputs[gid] * weights[gid];
}