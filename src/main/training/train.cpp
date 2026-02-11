/*
 * train
 * This file contains functions to facilitate the training of models for
 * anomaly detection.
 */

// OpenCV2
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Fault Sense
#include "../feature/feature-extraction.h"
#include "../feature/utils/pre-processing-utils.h"
#include "../feature/utils/generic-utils.h"
#include "../feature/objects/CannyThreshold.h"
// Standard
#include <map>
#include <array>
#include <string>
#include <cstdlib>
#include <cmath>

void trainMatrix(std::map<std::string, cv::Mat> &matrixNorm);
void trainCellNorms(std::map<std::string, std::array<float, 5>> &cellNorms, const bool normal, const std::string& category);
void trainCellNorm(std::map<std::string, std::array<float, 5>> &cellNorms, std::map<std::string, cv::Mat> &images, const std::string& categoryName);
void initNormMatrix(const std::map<std::string, cv::Mat>::iterator &itterator, int cellSize, cv::Mat &categoryNorm);
void updateCategoryNorm(cv::Mat norm, cv::Mat values, int cellSize, int numberOfSamples);

const std::string dataRoot = "../../../data/";
const std::string objectCategories[12] = {
    "chewinggum/",
    "candle/",
    "capsules/",
    "cashew/",
    "fryum/",
    "macaroni1/",
    "macaroni2/",
    "pcb1/",
    "pcb2/",
    "pcb3/",
    "pcb4/",
    "pipe_fryum/"
};
const std::string anomalyPath = "Data/Images/Anomaly";
const std::string normalPath = "Data/Images/Normal";
const int cellSize = 60; // shoulde be divisible by 2

/*
 * trainMatrix
 */
void trainMatrix(std::map<std::string, cv::Mat> &matrixNorm) {


    for (int i = 0; i < 12; i++) {
        std::map<std::string, cv::Mat> images = readImagesFromDirectory(dataRoot + objectCategories[i] + anomalyPath); 

        cv::Mat categoryNorm;
        initNormMatrix(images.begin(), cellSize, categoryNorm);

        for (const auto& [imageName, image] : images) {

            // Compute LBP values for each pixel
            cv::Mat LBPValues;
            lbpValues(image, LBPValues);

            updateCategoryNorm(categoryNorm, LBPValues, cellSize, images.size());
        }

        std::string objectCategory = objectCategories[i];
        objectCategory.pop_back(); // remove '/'
        std::cout << "objectCategory: " << objectCategory << std::endl;
        matrixNorm.insert({objectCategory, categoryNorm});
    }
}

/*
 * initNormMatrix
 * Initalizes an objects norm matrix
 * 
 * @param itterator std::map<std::string, cv::Mat>::iterator
 * @param cellSize int
 * @param categoryNorm cv::Mat
 */
void initNormMatrix(const std::map<std::string, cv::Mat>::iterator &itterator, int cellSize, cv::Mat &categoryNorm) {

    cv::Mat sampleImage = itterator->second; cv::Mat sampleValues;
    lbpValues(sampleImage, sampleValues); // To remove 1 pixel margin

    int rowMargin = sampleValues.rows % cellSize;
    int colMargin = sampleValues.cols % cellSize;

    categoryNorm = cv::Mat::zeros((sampleValues.rows - rowMargin) / cellSize, (sampleValues.cols - colMargin) / cellSize, CV_32FC(5));
}

/*
 * updateCategoryNorm
 * Updates a norm for a collection of samples with a sample
 *
 * @param norm (cv::Mat) The norm that will be updated
 * @param values (cv::Mat) The sample to update the norm with
 * @param cellSize (int) The size of elements in a sample 
 * @param numberOfSamples (int) The number of samples that will be used to create the norm
 */
void updateCategoryNorm(cv::Mat norm, cv::Mat values, int cellSize, int numberOfSamples) {

    int rowMargin = values.rows % cellSize;
    int colMargin = values.cols % cellSize;

    int collIndex, rowIndex = 0; 
    for (int row = rowMargin / 2; row < values.rows - (rowMargin / 2); row += cellSize) {
        collIndex = 0;
        for (int col = colMargin / 2; col < values.cols - (colMargin / 2); col += cellSize) {

            // Get LBP distribution of cell
            cv::Mat cell = values(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
            std::array<float, 5> LBPHistogram = {};
            lbpValueDistribution(cell, LBPHistogram);

            // Update the cell norm with the samples value
            float* cellDistribution = norm.ptr<float>(rowIndex,collIndex);
            for (int j = 0; j < 5; j++)
                cellDistribution[j] += LBPHistogram[j] / numberOfSamples;


            collIndex++;
        }
        rowIndex++;
    }
}

/*
 * trainCell
 *
 * @param cellNorms A mapping of object type (string) to it's average anomaly distribution
 * @param normal
 * @param category
 */
void trainCellNorms(std::map<std::string, std::array<float, 5>> &cellNorms, const bool normal, const std::string& category = "") {


    // Helper lambda to process images from a given path
    auto processCategory = [&](const std::string& categoryName) {
        const std::string imagePath = dataRoot + categoryName + "/" + (normal ? normalPath : anomalyPath);
        std::map<std::string, cv::Mat> images = readImagesFromDirectory(imagePath);
        trainCellNorm(cellNorms, images, categoryName);
    };
    
    // If category specified
    if (!category.empty()) {
        processCategory(category);

    // Else train all categories
    } else {
        for (const auto& category : objectCategories)
            processCategory(category);
    }
}


/*
 * trainCellNorm
 *
 * @param cellNorms 
 * @param images 
 */
void trainCellNorm(std::map<std::string, std::array<float, 5>> &cellNorms, std::map<std::string, cv::Mat> &images, const std::string& categoryName) {


    std::array<float, 5> averageLBPDistribution = {0};
    int totalAnomalyCells = 0;

    for (const auto& [imageName, image] : images) {

        // Get image mask (to identify anomaly cells)
        std::string maskName = imageName.substr(0, imageName.size() - 3) + "png";
        cv::Mat imageMask = cv::imread(dataRoot + "masks/" + categoryName + "/" + maskName);

        if (imageMask.empty()) {
            std::cerr << "Warning: Could not load mask for " << imageName << "\n";
            continue;
        }

        // Apply pre-processing to image
        cv::Mat LBPValues;
        lbpValues(image, LBPValues);

        // Accumulate the distribution of pixel values accross anomaly cells
        for (int row = cellSize / 2; row + cellSize < LBPValues.rows; row += cellSize) {
            for (int col = cellSize / 2; col + cellSize < LBPValues.cols; col += cellSize) {

                // Note we (+ 1) as LBPValues loses a pixel on each edge as part of it's construction
                if (checkIfCellIsNormal(imageMask(cv::Range(row + 1, row + cellSize), cv::Range(col + 1, col + cellSize)))) 
                    continue; // skip normal cells
                totalAnomalyCells++;

                // Get distribution of pixel values in anomaly cell
                std::array<float, 5> LBPHistogram = {0};
                cv::Mat cell = LBPValues(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
                lbpValueDistribution(cell, LBPHistogram);

                // Add to cumulative result
                for (int j = 0; j < 5; j++)
                    averageLBPDistribution[j] += LBPHistogram[j];
            }
        }

        // Average the cummulated distribution for anomaly cells
        if (totalAnomalyCells == 0) {
            std::cout << "totalAnomalyCells == 0\n";
            continue; // Need to investigate why this is happening
        }

    }

    if (totalAnomalyCells == 0) {
        std::cerr << "Warning: No anomaly cells found for category " 
                  << categoryName << std::endl;
        return;
    }

     // Average the cumulative distribution across all anomaly cells
    for (int k = 0; k < 5; k++) {
        averageLBPDistribution[k] /= totalAnomalyCells;
        averageLBPDistribution[k] /= images.size();
    }
    
    // Normalize to sum to 100
    float sum = 0.0f;
    for (int k = 0; k < 5; k++) {
        sum += averageLBPDistribution[k];
    }
    
    if (sum > 0.0f) {
        for (int k = 0; k < 5; k++) {
            averageLBPDistribution[k] = (averageLBPDistribution[k] / sum) * 100.0f;
        }
    }
    // Record result
    cellNorms.insert({categoryName, averageLBPDistribution});


    /*
    for (int l = 0; l < averageLBPDistribution.size(); l++)
        averageLBPDistribution[l] = (averageLBPDistribution[l] / images.size()) * 100;

    cellNorms.insert({categoryName, averageLBPDistribution});
    */
}
