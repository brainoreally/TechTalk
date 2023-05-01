#include <chrono>  // for timing
#include <thread>  // for sleeping
#include "src/Graphics/Graphics.h"
#include "src/NeurtalNetwork/NeuralNetwork.h"
#include "src/MNISTLoader/MNISTLoader.h"
#include "src/NeurtalNetwork/CLProgram/CLProgram.h"

NetworkParams buildPerceptronParams() {
    NetworkParams out = NetworkParams();

    out.kernel_source_path = "src/kernels/perceptronNetwork.cl";

    out.kernel_params = {
        { "dot_product_forward_pass", { "layerDepth", "valueOffsets", "layerValues", "inOutValues", "weightedValues", "networkValues" } },
        { "sigmoid_activation", { "layerDepth", "layerInputSize", "layerValues", "inOutValues" } },
        { "sigmoid_train_start", { "layerDepth", "valueOffsets", "derivitiveOuts", "inOutValues", "correctOutput" } },
        { "sigmoid_train_continue", { "layerError", "layerDepth", "valueOffsets", "networkValues", "derivitiveOuts" } },
        { "sigmoid_learn", { "layerDepth", "valueOffsets", "weightedValues", "networkValues", "derivitiveOuts" } },
        { "relu_activation", { "layerDepth", "layerInputSize", "layerValues", "inOutValues" } },
        { "perceptron_learn", { "layerDepth", "valueOffsets", "networkValues", "weightedValues", "inOutValues", "correctOutput" } },
    };

    std::map<const char*, const char*> inputLayerKernelKeys = {
        { "forward_pass", "dot_product_forward_pass" },
        { "train", "sigmoid_train_continue" },
        { "learn", "sigmoid_learn" },
    };

    std::map<const char*, const char*> hiddenLayerKernelKeys = {
        { "forward_pass", "dot_product_forward_pass" },
        { "activate", "sigmoid_activation" },
        { "train", "sigmoid_train_continue" },
        { "learn", "sigmoid_learn" },
    };

    std::map<const char*, const char*> outputLayerKernelKeys = {
        { "activate", "sigmoid_activation" },
        { "train", "sigmoid_train_start" },
    };

    out.valueOffsets = { };
    out.layerInputSize = { };

    int inputLayerSize = 2;
    std::vector<int> hiddenLayerSize = { 4, 3, };
    int outputLayersize = 1;

    int maxValues = inputLayerSize + 1;

    if (outputLayersize > maxValues)
        maxValues = outputLayersize;

    out.valueOffsets.push_back(0);
    out.layerInputSize.push_back(inputLayerSize);

    out.inputLayerParams = LayerParams(inputLayerSize, 1, 1, inputLayerKernelKeys);

    int totalNumNetworkValues = inputLayerSize + 1;
    int totalWeightedValues = totalNumNetworkValues;
    out.valueOffsets.push_back(totalNumNetworkValues);
    out.layerInputSize.push_back(inputLayerSize + 1);

    int previousLayerSize = totalWeightedValues;
    for (int size : hiddenLayerSize)
    {
        LayerParams hiddenP = LayerParams(size, 1, 1, hiddenLayerKernelKeys);

        out.hiddenLayerParams.push_back(hiddenP);
        totalNumNetworkValues += size + 1;
        totalWeightedValues += (size + 1) * previousLayerSize;
        previousLayerSize = size + 1;
        out.valueOffsets.push_back(totalNumNetworkValues);
        out.layerInputSize.push_back(size + 1);
        if (size + 1 > maxValues)
            maxValues = size + 1;
    }
    
    out.outputLayerParams = LayerParams(outputLayersize, 1, 1, outputLayerKernelKeys);

    out.totalNumOutputs = totalNumNetworkValues + outputLayersize;
    out.totalNumWeights = totalWeightedValues;
    out.maxValuesInLayer = maxValues;
    out.outputLayersize = outputLayersize;
    out.numLayersInNetwork = out.hiddenLayerParams.size() + 2;

    return out;
}

void loadMNISTData()
{
    std::string input_path = "MNIST/";
    std::string training_images_filepath = input_path + "train-images-idx3-ubyte/train-images-idx3-ubyte";
    std::string training_labels_filepath = input_path + "train-labels-idx1-ubyte/train-labels-idx1-ubyte";
    std::string test_images_filepath = input_path + "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    std::string test_labels_filepath = input_path + "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";
    MnistDataloader mnistData = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath);
    auto wat = mnistData.load_data();
}

int main() {
    Graphics graphics = Graphics();

    NetworkParams perceptronNetworkParams = buildPerceptronParams();

    NeuralNetwork<float> network = NeuralNetwork<float>(perceptronNetworkParams);

    std::vector<std::vector<float>> inputs = {
            { 0.0f, 0.0f },
            { 1.0f, 0.0f },
            { 0.0f, 1.0f },
            { 1.0f, 1.0f },
    };

    std::vector<float> outputs = {
        0.0f,
        1.0f,
        1.0f,
        1.0f
    };

    std::pair<std::vector<std::vector<float>>, std::vector<float>> trainingData = std::make_pair(inputs, outputs);

    GLuint iterations = 5000;
    network.train(trainingData, iterations, iterations / 10);

    int iter = 0;
    auto last_prediction_time = std::chrono::high_resolution_clock::now();  // initialize timer

    // Main loop
    while (graphics.is_running()) {
        // Check if 0.5 seconds have passed since last prediction
        auto now = std::chrono::high_resolution_clock::now();
        
        if (!network.training && std::chrono::duration_cast<std::chrono::milliseconds>(now - last_prediction_time).count() >= 500) {
            network.predict(trainingData.first[iter]);
            iter = (iter + 1) % 4;
            last_prediction_time = now;  // update last prediction time
        }
        
        graphics.setupScene();

        graphics.drawNeurons(network.returnNetworkValues());

        graphics.swapBuffersAndPoll();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(16));  // limit frame rate to ~60 fps
    }

    // Clean up
    return 0;
}
