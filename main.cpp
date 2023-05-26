#include <chrono>  // for timing
#include <thread>  // for sleeping
#include "src/Graphics/Graphics.h"
#include "src/NeurtalNetwork/NeuralNetwork.h"
#include "src/MNISTLoader/MNISTLoader.h"
#include "src/NeurtalNetwork/CLProgram/CLProgram.h"

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> convert(std::pair<std::vector<std::vector<uint8_t>>, std::vector<std::vector<float>>> in) {
    std::vector<std::vector<float>> floatInput;

    // Convert input values to floats between 0 and 1
    for (const auto& row : in.first) {
        std::vector<float> floatRow;
        for (const auto& value : row) {
            float floatValue = static_cast<float>(value) / 255.0f;
            floatRow.push_back(floatValue);
        }
        floatInput.push_back(floatRow);
    }

    return std::make_pair(floatInput, in.second);
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> loadMNISTData()
{
    std::string input_path = "MNIST/";
    std::string training_images_filepath = input_path + "train-images-idx3-ubyte/train-images-idx3-ubyte";
    std::string training_labels_filepath = input_path + "train-labels-idx1-ubyte/train-labels-idx1-ubyte";
    std::string test_images_filepath = input_path + "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    std::string test_labels_filepath = input_path + "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";
    MnistDataloader mnistData = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath);

    auto test = mnistData.load_data_f();
    return convert(test);// mnistData.load_data_f();
}

int main() {
    Graphics graphics = Graphics();

    std::vector<std::vector<float>> inputs = {
            { 0.0f, 0.0f },
            { 1.0f, 0.0f },
            { 0.0f, 1.0f },
            { 1.0f, 1.0f },
    };

    std::vector<std::vector<float>> outputs = {
       { 0.0f },
       { 1.0f },
       { 1.0f },
       { 1.0f }
    };

    NetworkParams perceptronNetworkParams = NetworkParams(
        "src/kernels/perceptron.cl",
        inputs[0].size(),
        outputs[0].size(),
        0,
        { { 0, { 4, 4, } }, },
        inputs.size()
    );
    int batchSize = 4;
    float learningRate = 1.0f;
    NeuralNetwork<float> network = NeuralNetwork<float>(perceptronNetworkParams);
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> trainingData = std::make_pair(inputs, outputs);

    /*
    NetworkParams mnistNetworkParams = buildMNISTParams();
    NeuralNetwork<float> network = NeuralNetwork<float>(mnistNetworkParams);
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> trainingData = loadMNISTData();
    */

    GLuint iterations = 50000;
    network.train(trainingData, iterations, iterations / 10, batchSize, learningRate);

    int iter = 0;
    auto last_prediction_time = std::chrono::high_resolution_clock::now();  // initialize timer

    // Main loop
    while (graphics.is_running()) {
        // Check if 0.5 seconds have passed since last prediction
        auto now = std::chrono::high_resolution_clock::now();
        
        if (!network.training && std::chrono::duration_cast<std::chrono::milliseconds>(now - last_prediction_time).count() >= 500) {
            network.predict(trainingData.first[iter], trainingData.second[iter]);
            iter = (iter + 1) % trainingData.second.size();
            last_prediction_time = now;  // update last prediction time
        }
        
        graphics.setupScene();

        graphics.drawNeurons(network.returnNetworkValues(), network.returnWeightValues(), network.returnBiasValues());


        graphics.swapBuffersAndPoll();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(16));  // limit frame rate to ~60 fps
    }

    // Clean up
    return 0;
}
