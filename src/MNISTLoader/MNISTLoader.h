#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <array>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <cstdio>

class MnistDataloader {
public:
    MnistDataloader(const std::string& training_images_filepath, const std::string& training_labels_filepath,
        const std::string& test_images_filepath, const std::string& test_labels_filepath)
        : training_images_filepath(training_images_filepath), training_labels_filepath(training_labels_filepath),
        test_images_filepath(test_images_filepath), test_labels_filepath(test_labels_filepath) {}

    std::pair<std::vector<std::vector<uint8_t>>, std::vector<uint8_t>> load_data() {
        auto train = read_images_labels(training_images_filepath, training_labels_filepath);
        auto test = read_images_labels(test_images_filepath, test_labels_filepath);
        return std::make_pair(std::move(train.first), std::move(train.second));
    }

private:
    std::string training_images_filepath;
    std::string training_labels_filepath;
    std::string test_images_filepath;
    std::string test_labels_filepath;

    inline uint32_t bswap32(uint32_t x) {
        return ((x & 0xFF000000) >> 24) |
            ((x & 0x00FF0000) >> 8) |
            ((x & 0x0000FF00) << 8) |
            ((x & 0x000000FF) << 24);
    }

    std::pair<std::vector<std::vector<uint8_t>>, std::vector<uint8_t>>
        read_images_labels(const std::string& images_filepath, const std::string& labels_filepath) {
            std::vector<uint8_t> labels;
            std::ifstream file(labels_filepath, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file: " + labels_filepath);
            }
            uint32_t magic_number, label_count;
            file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
            file.read(reinterpret_cast<char*>(&label_count), sizeof(label_count));
            magic_number = bswap32(magic_number);
            label_count = bswap32(label_count);
            if (magic_number != 2049) {
                throw std::runtime_error("Invalid magic number for labels file: " + std::to_string(magic_number));
            }
            labels.resize(label_count);
            file.read(reinterpret_cast<char*>(labels.data()), label_count);
            file.close();

            std::vector<std::vector<uint8_t>> images;
            file.open(images_filepath, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file: " + images_filepath);
            }
            uint32_t image_magic_number, image_count, rows, cols;
            file.read(reinterpret_cast<char*>(&image_magic_number), sizeof(image_magic_number));
            file.read(reinterpret_cast<char*>(&image_count), sizeof(image_count));
            file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
            image_magic_number = bswap32(image_magic_number);
            image_count = bswap32(image_count);
            rows = bswap32(rows);
            cols = bswap32(cols);
            if (image_magic_number != 2051) {
                throw std::runtime_error("Invalid magic number for images file: " + std::to_string(image_magic_number));
            }
            images.resize(image_count, std::vector<uint8_t>(rows * cols));
            for (size_t i = 0; i < image_count; ++i) {
                file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
            }
            file.close();

            return std::make_pair(std::move(images), std::move(labels));
    }
};