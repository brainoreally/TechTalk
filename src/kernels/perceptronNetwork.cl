
__kernel void perceptron_learn(__global float* layer_inputs, __global float* weights, __global float* output, __global float* correctOutput)
{
    const int gid = get_global_id(0);

    float err = correctOutput[0] - output[0];
    float learningRate = 1.0f;

    weights[gid] += (err * layer_inputs[gid] * learningRate);
}

__kernel void sigmoid_activation(__global float* layer_outputs, __global float* weights, __global float* output)
{
    float unNormalizedOutput = (layer_outputs[0]) + (layer_outputs[1]) + (layer_outputs[2]);
    output[0] = 1.0f / (1.0f + exp(-unNormalizedOutput));
}

__kernel void dot_product_forward_pass(__global float* layer_inputs, __global float* layer_outputs, __global float* weights)
{
    const int gid = get_global_id(0);
    layer_outputs[gid] = layer_inputs[gid] * weights[gid];
}