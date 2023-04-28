#include <chrono>  // for timing
#include <thread>  // for sleeping
#include "src/Graphics/Graphics.h"
#include "src/NeurtalNetwork/NeuralNetwork.h"
#include "src/MNISTLoader/MNISTLoader.h"

NetworkParams buildPerceptronParams() {
    NetworkParams out = NetworkParams(2, 1, 1, 1);

    std::map<const char*, const char*> inputLayerKernelKeys = {
        { "forward_pass", "dot_product_forward_pass" },
        { "activate", "sigmoid_activation" },
        { "learn", "perceptron_learn" },
    };
    std::map<const char*, const char*> inputLayerBufferKeys = {
        { "inputs", "inputs" },
        { "weights", "weights" },
        { "correctOutput", "correctOutput" },
    };
    out.inputLayerParams = LayerParams(2, 1, 1, inputLayerKernelKeys, inputLayerBufferKeys);
    out.outputLayerParams = LayerParams(1, 1, 1, {}, { {"output", "output"} });

    return out;
}

void setupPerceptronCL() {
    CLProgram::createKernel("dot_product_forward_pass", new size_t[1]{ 2 }, new size_t[1]{ 1 });
    CLProgram::createKernel("sigmoid_activation", new size_t[1]{ 1 }, new size_t[1]{ 1 });
    CLProgram::createKernel("perceptron_learn", new size_t[1]{ 3 }, new size_t[1]{ 1 });

    CLProgram::createBuffer("inputs", 2);
    CLProgram::createBuffer("weights", 3);
    CLProgram::createBuffer("output", 1);
    CLProgram::createBuffer("correctOutput", 1);

    CLProgram::setKernelParam("dot_product_forward_pass", 0, "inputs");
    CLProgram::setKernelParam("dot_product_forward_pass", 1, "weights");

    CLProgram::setKernelParam("sigmoid_activation", 0, "inputs");
    CLProgram::setKernelParam("sigmoid_activation", 1, "weights");
    CLProgram::setKernelParam("sigmoid_activation", 2, "output");

    CLProgram::setKernelParam("perceptron_learn", 0, "inputs");
    CLProgram::setKernelParam("perceptron_learn", 1, "weights");
    CLProgram::setKernelParam("perceptron_learn", 2, "output");
    CLProgram::setKernelParam("perceptron_learn", 3, "correctOutput");
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
    NeuralNetwork network = NeuralNetwork(perceptronNetworkParams);

    std::vector<std::vector<std::vector<float>>> trainingData = {
        { { 0.0f, 0.0f }, { 0.0f }, },
        { { 1.0f, 0.0f }, { 1.0f }, },
        { { 0.0f, 1.0f }, { 1.0f }, },
        { { 1.0f, 1.0f }, { 1.0f }, },
    };

    GLuint iterations = 5000;
    network.train(trainingData, iterations, iterations / 10);

    int iter = 0;
    auto last_prediction_time = std::chrono::high_resolution_clock::now();  // initialize timer

    // Main loop
    while (graphics.is_running()) {

        // Check if 0.5 seconds have passed since last prediction
        auto now = std::chrono::high_resolution_clock::now();
        if (!network.training && std::chrono::duration_cast<std::chrono::milliseconds>(now - last_prediction_time).count() >= 500) {
            network.predict(trainingData[iter][0]);
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
