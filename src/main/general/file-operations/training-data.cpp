/*
 * training-data.cpp
 * A collection of functions for the reading and writing of training data to 
 * persistant memory
 */

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <map>
#include <array>
#include <string>
#include <iostream>
#include <fstream>
//Fault Sense
#include "../../../global-variables.h"

void writeCellDistributions(std::map<std::string, std::array<float, 5>> &distributions);
void readCellDistributions(std::map<std::string, std::array<float, 5>> &distributions);
void writeMatrixNorm(std::map<std::string, cv::Mat> &norms); 
void readMatrixNorm(std::map<std::string, cv::Mat> &norms);

const std::string NORMAL_FILEPATH = "data/trained-data/normal-training-samples.yml";
const std::string ANOMALY_FILEPATH = "data/trained-data/anomaly-training-samples.yml";

/*
 * writeCellDistributions
 * Writes a set of pixel value distribtutions for given samples to a yaml file
 *
 * @param distributions (std::map<std::string, std::array<float, 5>>) The collection of distributions to be written to a yaml
 */
void writeCellDistributions(std::map<std::string, std::array<float, 5>> &distributions) {

    std::ofstream file(global::projectRoot + ANOMALY_FILEPATH);

    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + global::projectRoot + ANOMALY_FILEPATH);

    for (const auto& [category, distribution] : distributions) {
        file << category << ":\n";
        for (int index = 0; index < 5; index++)
            file << "  - " << std::fixed << std::setprecision(6) << distribution[index] << "\n";
    }

    file.close();
}

/*
 * readCellDistributions
 * Reads a set of pixel value distribtutions for given samples from a yaml file
 *
 * @param distributions (std::map<std::string, std::array<float, 5>>) The collection of distributions to be written to a yaml
 */
void readCellDistributions(std::map<std::string, std::array<float, 5>> &distributions) {

    std::ifstream file(global::projectRoot + ANOMALY_FILEPATH);

    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + global::projectRoot + ANOMALY_FILEPATH);
    
    std::cout << "INFO: reading cell distributions ...\n";

    int index = 5;
    std::array<float, 5> distribution;
    std::string category = "";
    std::string line;
    while (std::getline(file, line)) {

        if (line.empty() || line[0] == '#') continue;

        if (index == 5) {

            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find(":"));
            category = line;
            std::cout << "INFO: reading category: " << category << "\n";
            index = 0;

        } else {

            line.erase(0, line.find('-') + 2);
            line.erase(line.find_last_not_of(" \t") + 1);

            try {
                float value = std::stof(line);
                distribution[index] = value;
                index++;

                if (index == 5) {
                    distributions[category] = distribution;
                }
            } catch (std::exception &exception) {
                std::cout << "exception: " << exception.what() << "\n";
            }
        }
    }
    distributions.insert( {category, distribution} );
    file.close();
}

/*
 * writeMatrixNorm
 * Writes a collection of matrixes of norm distributions to a yaml file 
 *
 * @param norms (std::map< std::string, cv::Mat>) The norm matrices to be written to yaml
 */
void writeMatrixNorm(std::map<std::string, cv::Mat> &norms) {

    cv::FileStorage fs(global::projectRoot + NORMAL_FILEPATH, cv::FileStorage::WRITE);

    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open file for writing: " + global::projectRoot + NORMAL_FILEPATH);
    }
    
    std::cout << "INFO: writing matrix norms ...\n";
    
    for (const auto& [category, matrixNorm] : norms) {
        std::cout << "INFO: writing category: " << category << "\n";
        fs << category << matrixNorm;
    }

    fs.release();
}

/*
 * readMatrixNorm
 * Reads a collection of matrixes of norm distributions from a yaml file 
 *
 * @param norms (std::map< std::string, cv::Mat>) The norm matrix categories to be populated 
 */
void readMatrixNorm(std::map<std::string, cv::Mat> &norms) {

    cv::FileStorage fs(global::projectRoot + NORMAL_FILEPATH, cv::FileStorage::READ);

    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open file for reading: " + global::projectRoot + NORMAL_FILEPATH);
    }
    std::cout << "INFO: reading matrix norms ...\n";
    
    for (auto& [category, matrixNorm] : norms) {
        std::cout << "INFO: reading category: " << category << "\n";
        fs[category] >> matrixNorm;
    }

    fs.release();
}
