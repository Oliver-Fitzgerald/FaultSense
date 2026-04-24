/*
 * feature-file-operations.cpp
 * A collection of functions for the reading and writing of trained features to 
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


/*
 * writeObjectFeatures
 * Writes a collection of feature matrixes to a yaml file for an object category
 * Note: A duplicate object category will be overriden
 * @param features          - The features for the object category
 * @param objectCategory    - The object category for which features are being written
 * @param normal            - Indicates wether the features where trained on normal or anomaly samples
 */
void writeObjectFeatures(std::map<std::string, cv::Mat> &features, const std::string objectCategory, bool normal) {

    
    std::string filePath;
    if (normal)
        filePath = "data/trained-data/" + objectCategory + "-normal-features.yml";
    else
        filePath = "data/trained-data/" + objectCategory + "-anomaly-features.yml";

    cv::FileStorage fs(global::projectRoot + filePath, cv::FileStorage::WRITE);

    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open file for writing: " + global::projectRoot + filePath);
    }
    
    for (const auto& [feature, matrixNorm] : features) {
        std::cout << "INFO: writing features to file : " << filePath << "\n";
        fs << feature << matrixNorm;
    }

    fs.release();
}

/*
 * readObjectFeatures
 * Reads a collection of matrixes of norm distributions from a yaml file 
 *
 * @param features       - The data structure to store feature matrices
 * @param objectCategory - The object category for which features should be retreived
 * @param normal         - Indicates wether normal or anoamly samples should be retreived
 */
void readObjectFeatures(std::map<std::string, cv::Mat> &features, const std::string objectCategory, bool normal) {

    std::string filePath;
    if (normal)
        filePath = "data/trained-data/" + objectCategory + "-normal-features.yml";
    else
        filePath = "data/trained-data/" + objectCategory + "-anomaly-features.yml";

    cv::FileStorage fs(global::projectRoot + filePath, cv::FileStorage::READ);

    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open file for reading: " + global::projectRoot + filePath);
    }

    for (auto& [feature, featureMatrix] : features) {
        std::cout << "INFO: reading features from file : " << global::projectRoot << filePath << "\n";
        std::cout << featureMatrix << "\n";
        fs[feature] >> featureMatrix;
    }

    fs.release();
}
