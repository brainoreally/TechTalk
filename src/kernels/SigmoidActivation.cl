float sigmoidActivation(float output, float bias)
{
    output = 1 / (1 + exp(-output));
    return output;
}

__kernel void activate(__global float* inputs, __global float* weights, __global float* output)
{
    float bias = 1.0f;
    float unNormalizedOutput = (inputs[0]) + (inputs[1]) + (bias * weights[2]);
    output[0] = sigmoidActivation(unNormalizedOutput, bias);
}