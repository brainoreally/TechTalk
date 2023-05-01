#include <chrono>  // for timing
#include <thread>  // for sleeping
#include "src/Graphics/Graphics.h"
#include "src/NeurtalNetwork/NeuralNetwork.h"
#include "src/MNISTLoader/MNISTLoader.h"
#include "src/NeurtalNetwork/CLProgram/CLProgram.h"

NetworkParams buildPerceptronParams() {
    NetworkParams out = NetworkParams();

    std::map<const char*, const char*> inputLayerKernelKeys = {
        { "forward_pass", "dot_product_forward_pass" },
        { "learn", "perceptron_learn" },
    };

    std::map<const char*, const char*> outputLayerKernelKeys = {
        { "activate", "sigmoid_activation" },
    };


    out.inputLayerParams = LayerParams(2, 1, 1, inputLayerKernelKeys);
    out.outputLayerParams = LayerParams(1, 1, 1, outputLayerKernelKeys);

    return out;
}

void setupPerceptronCL() {
    CLProgram::createKernel("dot_product_forward_pass");
    CLProgram::createKernel("sigmoid_activation");
    CLProgram::createKernel("perceptron_learn");
    CLProgram::createKernel("reset_depth");
    CLProgram::createKernel("advance_layer");
    CLProgram::createKernel("add_outputs_to_network_values");

    int totalNumOutputs = 1;
    int totalNumWeights = 2 + 1; // We have 2 inputs and 1 bias; so our network will want parallel jobs for this many values
    int maxValuesInLayer = 3;
    int outputLayersize = 1;
    int numLayersInNetwork = 2;
    CLProgram::createBuffer<float>("networkValues", totalNumWeights + totalNumOutputs); // float collection - #neurons in network + biases
    CLProgram::createBuffer<float>("weightedValues", totalNumWeights); // float collection - #values with a weight (inputs on layers + bias)
    CLProgram::createBuffer<float>("inOutValues", maxValuesInLayer); // float collection - max #values in a layer output (input layer doesn't count; our network has just 1 output layer value)
    CLProgram::createBuffer<float>("layerValues", maxValuesInLayer); // float collection - max #values in a layer output (input layer doesn't count; our network has just 1 output layer value)
    CLProgram::createBuffer<float>("correctOutput", outputLayersize); // float collection - should match the output layer of our network in size
    CLProgram::createBuffer<unsigned int>("layerDepth", 1); // single int - used by network to track current layer index
    CLProgram::createBuffer<unsigned int>("valueOffsets", numLayersInNetwork); // collection of unsigned ints - used to track offsets for beginning of layers data in networkValues dataset

    CLProgram::setKernelParam("reset_depth", 0, "layerDepth");

    CLProgram::setKernelParam("dot_product_forward_pass", 0, "layerDepth");
    CLProgram::setKernelParam("dot_product_forward_pass", 1, "valueOffsets");
    CLProgram::setKernelParam("dot_product_forward_pass", 2, "layerValues");
    CLProgram::setKernelParam("dot_product_forward_pass", 3, "inOutValues");
    CLProgram::setKernelParam("dot_product_forward_pass", 4, "weightedValues");
    CLProgram::setKernelParam("dot_product_forward_pass", 5, "networkValues");

    CLProgram::setKernelParam("advance_layer", 0, "layerDepth");

    CLProgram::setKernelParam("sigmoid_activation", 0, "layerDepth");
    CLProgram::setKernelParam("sigmoid_activation", 1, "valueOffsets");
    CLProgram::setKernelParam("sigmoid_activation", 2, "layerValues");
    CLProgram::setKernelParam("sigmoid_activation", 3, "inOutValues");

    CLProgram::setKernelParam("perceptron_learn", 0, "layerDepth");
    CLProgram::setKernelParam("perceptron_learn", 1, "valueOffsets");
    CLProgram::setKernelParam("perceptron_learn", 2, "networkValues");
    CLProgram::setKernelParam("perceptron_learn", 3, "weightedValues");
    CLProgram::setKernelParam("perceptron_learn", 4, "inOutValues");
    CLProgram::setKernelParam("perceptron_learn", 5, "correctOutput");

    CLProgram::setKernelParam("add_outputs_to_network_values", 0, "layerDepth");
    CLProgram::setKernelParam("add_outputs_to_network_values", 1, "valueOffsets");
    CLProgram::setKernelParam("add_outputs_to_network_values", 2, "networkValues");
    CLProgram::setKernelParam("add_outputs_to_network_values", 3, "inOutValues");
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

    const char* perceptron_kernel_source_path = "src/kernels/perceptronNetwork.cl";
    CLProgram::initCL(perceptron_kernel_source_path);
    setupPerceptronCL();

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
