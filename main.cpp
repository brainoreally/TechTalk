#include <chrono>  // for timing
#include <thread>  // for sleeping
#include "src/Graphics/Graphics.h"
#include "src/NeurtalNetwork/NeuralNetwork.h"
#include "src/MNISTLoader/MNISTLoader.h"
#include "src/NeurtalNetwork/CLProgram/CLProgram.h"

NetworkParams buildMNISTParams() {
    NetworkParams out = NetworkParams();

    out.kernel_source_path = "src/kernels/perceptron.cl";
    std::vector<std::vector<int>> hiddenLayerParams = { { 64, 0 }, { 32, 0 }, { 16, 0 }, { 8, 0 } };
    out.numSamples = 60000;
    out.numInputs = 784;
    out.layerSizes = { out.numInputs };
    out.layerActivations = { 0 };
    out.numOutputs = 1;

    out.inputLayerParams = LayerParams(out.numInputs, 1, 1);

    out.numNeurons = out.numInputs;
    out.numWeights = 0;
    out.numLayers = 1;

    int previousLayerSize = out.numInputs;
    for (auto data : hiddenLayerParams)
    {
        LayerParams hiddenP = LayerParams(data[0], 1, 1);

        out.hiddenLayerParams.push_back(hiddenP);
        out.layerSizes.push_back(data[0]);
        out.layerActivations.push_back(data[1]);

        out.numNeurons += data[0];
        out.numWeights += data[0] * previousLayerSize;
        ++out.numLayers;
        previousLayerSize = data[0];
    }

    ++out.numLayers;
    out.layerSizes.push_back(out.numOutputs);
    out.numWeights += out.numOutputs * previousLayerSize;
    out.numNeurons += out.numOutputs;
    out.outputLayerParams = LayerParams(out.numOutputs, 1, 1);

    return out;
}

NetworkParams buildPerceptronParams() {
    NetworkParams out = NetworkParams();

    out.kernel_source_path = "src/kernels/perceptron.cl";
    std::vector<std::vector<int>> hiddenLayerParams = { { 5, 0 }, { 4, 0 }, { 2, 0 }, };
    out.numSamples = 4;
    out.numInputs = 2;
    out.layerSizes = { 2 };
    out.layerActivations = { 1 };
    out.numOutputs = 1;

    out.inputLayerParams = LayerParams(out.numInputs, 1, 1);

    out.numNeurons = out.numInputs;
    out.numWeights = 0;
    out.numLayers = 1;

    int previousLayerSize = out.numInputs;
    for (auto data : hiddenLayerParams)
    {
        LayerParams hiddenP = LayerParams(data[0], 1, 1);

        out.hiddenLayerParams.push_back(hiddenP);
        out.layerSizes.push_back(data[0]);
        out.layerActivations.push_back(data[1]);

        out.numNeurons += data[0];
        out.numWeights += data[0] * previousLayerSize;
        ++out.numLayers;
        previousLayerSize = data[0];
    }

    ++out.numLayers;
    out.layerSizes.push_back(out.numOutputs);
    out.numWeights += out.numOutputs * previousLayerSize;
    out.numNeurons += out.numOutputs;
    out.outputLayerParams = LayerParams(out.numOutputs, 1, 1);

    return out;
}


std::pair<std::vector<std::vector<float>>, std::vector<float>> convert(std::pair<std::vector<std::vector<uint8_t>>, std::vector<uint8_t>> in) {
    std::vector<std::vector<float>> floatInput;
    std::vector<float> floatOutput;

    // Convert input values to floats between 0 and 1
    for (const auto& row : in.first) {
        std::vector<float> floatRow;
        for (const auto& value : row) {
            float floatValue = static_cast<float>(value) / 255.0f;
            floatRow.push_back(floatValue);
        }
        floatInput.push_back(floatRow);
    }

    // Convert output values to floats between 0 and 1
    for (const auto& value : in.second) {
        float floatValue = static_cast<float>(value) / 255.0f;
        floatOutput.push_back(floatValue);
    }

    return std::make_pair(floatInput, floatOutput);
}

std::pair<std::vector<std::vector<float>>, std::vector<float>> loadMNISTData()
{
    std::string input_path = "MNIST/";
    std::string training_images_filepath = input_path + "train-images-idx3-ubyte/train-images-idx3-ubyte";
    std::string training_labels_filepath = input_path + "train-labels-idx1-ubyte/train-labels-idx1-ubyte";
    std::string test_images_filepath = input_path + "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    std::string test_labels_filepath = input_path + "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";
    MnistDataloader mnistData = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath);

    auto test = mnistData.load_data();
    return convert(test);// mnistData.load_data_f();
}

int main() {
    Graphics graphics = Graphics();
    /*
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
    */
    NetworkParams mnistNetworkParams = buildMNISTParams();
    NeuralNetwork<float> network = NeuralNetwork<float>(mnistNetworkParams);
    std::pair<std::vector<std::vector<float>>, std::vector<float>> trainingData = loadMNISTData();
    
    GLuint iterations = 1000;
    network.train(trainingData, iterations, iterations / 100);

    int iter = 0;
    auto last_prediction_time = std::chrono::high_resolution_clock::now();  // initialize timer

    // Main loop
    while (graphics.is_running()) {
        // Check if 0.5 seconds have passed since last prediction
        auto now = std::chrono::high_resolution_clock::now();
        
        if (!network.training && std::chrono::duration_cast<std::chrono::milliseconds>(now - last_prediction_time).count() >= 500) {
            network.predict(trainingData.first[iter]);
            iter = (iter + 1) % trainingData.second.size();
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
