float sigmoidActivation(float output)
{
    output = 1.0f / (1.0f + exp(-output));
    return output;
}

__kernel void learn(__global float* inputs, __global float* weights, __global float* output, __global float* correctOutput)
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

__kernel void activate(__global float* inputs, __global float* weights, __global float* output)
{
    float bias = 1.0f;
    float unNormalizedOutput = (inputs[0]) + (inputs[1]) + (bias * weights[2]);
    output[0] = sigmoidActivation(unNormalizedOutput);
}

__kernel void forward_pass(__global float* inputs, __global float* weights)
{
    const int gid = get_global_id(0);
    inputs[gid] = inputs[gid] * weights[gid];
}