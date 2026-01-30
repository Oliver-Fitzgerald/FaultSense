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

void writeDistributions(std::map<std::string, std::array<float, 5>> &distributions);
void readDistributions(std::map<std::string, std::array<float, 5>> &distributions);
void writeNorm(std::map<std::string, cv::Mat> &norms); 
void readNorm(std::map<std::string, cv::Mat> &norms);

const std::string NORMAL_FILEPATH = "../../../data/trained-data/normal-training-samples.yml";
const std::string ANOMALY_FILEPATH = "../../../data/trained-data/anomaly-training-samples.yml";

/*
 * writeDistributions
 * Writes a set of pixel value distribtutions for given samples to a yaml file
 *
 * @param distributions (std::map<std::string, std::array<float, 5>>) The collection of distributions to be written to a yaml
 */
void writeDistributions(std::map<std::string, std::array<float, 5>> &distributions) {

    std::ofstream file(ANOMALY_FILEPATH);

    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + ANOMALY_FILEPATH);

    for (const auto& [category, distribution] : distributions) {
        file << category << ":\n";
        for (int index = 0; index < 5; index++)
            file << "  - " << std::fixed << std::setprecision(6) << distribution[index] << "\n";
    }

    file.close();
}

/*
 * readDistributions
 * Reads a set of pixel value distribtutions for given samples from a yaml file
 *
 * @param distributions (std::map<std::string, std::array<float, 5>>) The collection of distributions to be written to a yaml
 */
void readDistributions(std::map<std::string, std::array<float, 5>> &distributions) {

    std::ifstream file(ANOMALY_FILEPATH);

    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + ANOMALY_FILEPATH);

    int index = 5;
    std::array<float, 5> distribution;
    std::string category = "";
    std::string line;
    while (std::getline(file, line)) {

        if (line.empty() || line[0] == '#') continue;

        if (index == 5) {
            
            if (category != "")
                distributions.insert( {category, distribution} );

            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find(":"));
            category = line;
            index = 0;

        } else {

            line.erase(0, line.find('-') + 2);
            line.erase(line.find_last_not_of(" \t") + 1);

            try {
                float value = std::stof(line);
                distribution[index] = value;
                index++;
            } catch (std::exception &exception) {
                std::cout << "exception: " << exception.what() << "\n";
            }
        }
    }
    distributions.insert( {category, distribution} );
    file.close();
}

/*
 * writeNorm
 * Writes a collection of matrixes of norm distributions to a yaml file 
 *
 * @param norms (std::map< std::string, cv::Mat>) The norm matrices to be written to yaml
 */
void writeNorm(std::map<std::string, cv::Mat> &norms) {

    cv::FileStorage fs(NORMAL_FILEPATH, cv::FileStorage::WRITE);

    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open file for writing: " + NORMAL_FILEPATH);
    }
    
    for (const auto& [category, matrixNorm] : norms)
        fs << category << matrixNorm;

    fs.release();
}

/*
 * readNorm
 * Reads a collection of matrixes of norm distributions from a yaml file 
 *
 * @param norms (std::map< std::string, cv::Mat>) The norm matrix categories to be populated 
 */
void readNorm(std::map<std::string, cv::Mat> &norms) {

    cv::FileStorage fs(NORMAL_FILEPATH, cv::FileStorage::READ);

    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open file for reading: " + NORMAL_FILEPATH);
    }
    
    for (auto& [category, matrixNorm] : norms)
        fs[category] >> matrixNorm;

    fs.release();
}
