
__kernel void dot_product_forward_pass(
    __global unsigned int* layerDepth,
    __global unsigned int* valueOffsets,
    __global float* layerValues,
    __global float* inputValues,
    __global float* weightValues,
    __global float* networkValues
)
{
    int layerOffset = valueOffsets[layerDepth[0]];
    int rowID = get_global_id(0);
    int colID = get_global_id(1);
    int valID = rowID * colID;

    //printf("valueOffsets[layerDepth[0]] %f\n", valueOffsets[layerDepth[0]]);
    //printf("layerDepth[0] %f\n", layerDepth[0]);
    networkValues[valueOffsets[layerDepth[0]] + valID] = inputValues[rowID];

    //printf("networkValues[valueOffsets[layerDepth[0]] + gid]: %f \n", networkValues[valueOffsets[layerDepth[0]] + gid]);
    layerValues[gid] = inputValues[gid] * weightValues[+ valID];
    //printf("inputValues[gid]: %f \n", inputValues[gid]);
    //printf("weightValues[valueOffsets[layerDepth[0]] + gid]: %f \n", weightValues[valueOffsets[layerDepth[0]] + gid]);
    //printf("layerValues[gid]: %f \n", layerValues[gid]);
}

__kernel void reset_depth(
    __global unsigned int* layerDepth
) {
    //printf("before: %i\n", layerDepth[0]);
    layerDepth[0] = 0;
    //printf("after: %i\n", layerDepth[0]);
}

__kernel void advance_layer(
    __global unsigned int* layerDepth
) {
    //printf("before: %i\n", layerDepth[0]);
    layerDepth[0]++;
    //printf("after: %i\n", layerDepth[0]);
}

__kernel void decrease_layer(
    __global unsigned int* layerDepth
) {
    //printf("before: %i\n", layerDepth[0]);
    layerDepth[0]--;
    //printf("after: %i\n", layerDepth[0]);
}

__kernel void relu_activation(
    __global unsigned int* layerDepth,
    __global unsigned int* layerInputSize,
    __global float* layerValues,
    __global float* outputValues
)
{
    const int gid = get_global_id(0);

    float unNormalizedOutput = 0.0f;

    for (int i = 0; i < layerInputSize[layerDepth[0]]; i++)
        unNormalizedOutput += layerValues[i];

    outputValues[gid] = max(0.0f, unNormalizedOutput);
}

__kernel void sigmoid_activation(
    __global unsigned int* layerDepth,
    __global unsigned int* layerInputSize,
    __global float* layerValues,
    __global float* outputValues
)
{
    const int gid = get_global_id(0);

    float unNormalizedOutput = 0.0f;

    for(int i = 0; i < layerInputSize[layerDepth[0]]; i++)
        unNormalizedOutput += layerValues[i];

    outputValues[gid] = 1.0f / (1.0f + exp(-unNormalizedOutput));
    //printf("outputValues[gid] : %f \n", outputValues[gid]);
}

__kernel void sigmoid_train_start(
    __global unsigned int* layerDepth,
    __global unsigned int* valueOffsets,
    __global float* derivitiveOuts,
    __global float* outputValues,
    __global float* correctOutput
)
{
    //printf("===============begin - train start=============\n");
    const int gid = get_global_id(0);
    //printf("correctOutput[gid] : %f \n", correctOutput[gid]);
    //printf("outputValues[gid] : %f \n", outputValues[gid]);
    float err = correctOutput[gid] - outputValues[gid];
    //printf("err : %f \n", err);
    //printf("layerDepth[0]: %i\n", layerDepth[0]);
    //printf("valueOffsets[layerDepth[0]] + gid: %i\n", valueOffsets[layerDepth[0]] + gid);
    derivitiveOuts[valueOffsets[layerDepth[0]] + gid] = err * (outputValues[gid] / (1 - outputValues[gid]));
    //printf("derivitiveOuts[valueOffsets[layerDepth[0]] + gid] : %f \n", derivitiveOuts[valueOffsets[layerDepth[0]] + gid]);
    //printf("derivitiveOuts[3] : %f \n", derivitiveOuts[3]);
    //printf("===============end - train start=============\n");
}

__kernel void set_layer_error(
    __global float* layerError,
    __global unsigned int* layerDepth,
    __global unsigned int* valueOffsets,
    __global float* networkValues,
    __global float* derivitiveOuts
)
{
    const int lid = get_local_id(0);
    const int count = get_local_size(0);

    const int thisValueOffsets = valueOffsets[layerDepth[0]] + lid;
    const int nextLayerOffsets = valueOffsets[layerDepth[0] + 1] + lid;
    
    __local float errVal[3];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    errVal[lid] = networkValues[thisValueOffsets] * derivitiveOuts[nextLayerOffsets];
    //printf("errVal[lid]: %f\n", errVal[lid]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        layerError[0] = 0.0f;
        for (int i = 0; i < count; i++) {
            layerError[0] += errVal[i];
            //printf("layerError: %f\n", layerError[0]);
        }
    }
}

__kernel void sigmoid_train_continue(
    __global float* layerError,
    __global unsigned int* layerDepth,
    __global unsigned int* valueOffsets,
    __global float* networkValues,
    __global float* derivitiveOuts
)
{
    //printf("===============begin - train continue=============\n");
    const int gid = get_global_id(0);

    const int thisValueOffsets = valueOffsets[layerDepth[0]] + gid;

    derivitiveOuts[thisValueOffsets] = layerError[0] * (networkValues[thisValueOffsets] / (1 - networkValues[thisValueOffsets]));
    //printf("derivitiveOuts[thisLayerOffsets] : %f \n", derivitiveOuts[thisValueOffsets]);
    //printf("nextLayer : %i \n", nextLayerOffsets);
    //printf("derivitiveOuts[nextLayer] : %f \n", derivitiveOuts[nextLayerOffsets]);
    //printf("derivitiveOuts[3] : %f \n", derivitiveOuts[3]);
    //printf("===============end - train continue=============\n");
}

__kernel void sigmoid_learn(
    __global unsigned int* layerDepth,
    __global unsigned int* valueOffsets,
    __global float* weightValues,
    __global float* networkValues,
    __global float* derivitiveOuts
)
{
    const int gid = get_global_id(0);
    const int offset = valueOffsets[layerDepth[0]] + gid;

    //printf("offset: %i\n", offset);
    //printf("networkValues[offset]: %f\n", offset);
    //printf("derivitiveOuts[offset]: %f\n", offset);
    float learningRate = 0.1f;
    float weightAdjustment = max(0.0f, networkValues[offset] * derivitiveOuts[offset]) * learningRate;
    //printf("weightAdj: %f\n", weightAdjustment);
    weightValues[offset] -= weightAdjustment;
    //printf("multiple: %f\n", weightAdjustment);// networkValues[offset] * derivitiveOuts[offset]);
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
    float learningRate = 0.1f;
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
    //printf("networkValues: {%f, %f, %f}->{%f}\n", networkValues[0], networkValues[1], networkValues[2], networkValues[3]);
    /*printf("networkValues: {%f, %f, %f}->{%f, %f, %f, %f, %f}->{%f, %f, %f, %f, %f}->{%f}\n",
        networkValues[0], networkValues[1], networkValues[2],
        networkValues[3], networkValues[4], networkValues[5],
        networkValues[6], networkValues[7], networkValues[8],
        networkValues[9], networkValues[10], networkValues[11],
        networkValues[12], networkValues[13]);*/
}
