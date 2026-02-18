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
#include "../objects/CannyThreshold.h"
#include "../objects/PreProcessing.h"
#include "../general/file-operations/generic-file-operations.h"
#include "../general/generic-utils.h"
// Standard
#include <map>
#include <array>
#include <string>
#include <cstdlib>
#include <cmath>

namespace {

    const std::string dataRoot = "../data/";
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

    void initNormMatrix(const std::map<std::string, cv::Mat>::iterator &itterator, int cellSize, cv::Mat &categoryNorm);
    void initNormMatrix(const cv::Mat &sampleImage, int cellSize, cv::Mat &categoryNorm);
    void updateCategoryNorm(cv::Mat norm, cv::Mat values, int cellSize, int numberOfSamples);
    void generateNormalCellNorm(std::array<float, 5> &cellNorm, std::vector<cv::Mat> &images, const PreProcessing &preProcessingConfiguration);
    void generateAnomalyCellNorm(std::array<float, 5> &cellNorm, std::map<std::string, cv::Mat> &images, const PreProcessing &preProcessingConfiguration, const std::string &categoryName);
}

/*
 * trainMatrix
 * Description
 * @param matrixNorm
 * @param preProcessingConfiguration
 * @param normal
 */
void trainMatrix(std::map<std::string, cv::Mat> &matrixNorms, PreProcessing &preProcessingConfiguration, const bool normal) {


    for ( auto& [categoryName, categoryNorm] : matrixNorms ) {

        if (categoryName.empty()) continue;

        std::vector<cv::Mat> images;
        const std::string imagePath = dataRoot + categoryName + "/" + (normal ? normalPath : anomalyPath);
        readImagesFromDirectory(imagePath, images);

        initNormMatrix(images[0], cellSize, categoryNorm);

        for (auto& image : images) {

            // Compute LBP values for each pixel
            preProcessingConfiguration.apply(image);
            updateCategoryNorm(categoryNorm, image, cellSize, images.size());
        }
    }
}


/*
 * trainCellNorms
 * Description
 * @param cellNorms A mapping of object type (string) to it's average anomaly distribution
 * @param normal
 */
void trainCellNorms(std::map<std::string, std::array<float, 5>> &cellNorms, PreProcessing &preProcessingConfiguration, const bool normal) {


    for ( auto& [categoryName, categoryNorm] : cellNorms ) {

        if (categoryName.empty()) continue;

        const std::string imagePath = dataRoot + categoryName + "/" + (normal ? normalPath : anomalyPath);
        if (normal) {
            std::vector<cv::Mat> images;
            readImagesFromDirectory(imagePath, images);
            generateNormalCellNorm(categoryNorm, images, preProcessingConfiguration);

        } else {
            std::map<std::string, cv::Mat> images;
            readImagesFromDirectory(imagePath, images);
            generateAnomalyCellNorm(categoryNorm, images, preProcessingConfiguration, categoryName);
        }
    }
}



namespace {

    /*
     * initNormMatrix
     * Initalizes an objects norm matrix
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
     * initNormMatrix
     * Initalizes an objects norm matrix
     * @param itterator std::map<std::string, cv::Mat>::iterator
     * @param cellSize int
     * @param categoryNorm cv::Mat
     */
    void initNormMatrix(const cv::Mat &sampleImage, int cellSize, cv::Mat &categoryNorm) {

        int rowMargin = sampleImage.rows % cellSize;
        int colMargin = sampleImage.cols % cellSize;

        categoryNorm = cv::Mat::zeros((sampleImage.rows - rowMargin) / cellSize, (sampleImage.cols - colMargin) / cellSize, CV_32FC(5));
    }

    /*
     * updateCategoryNorm
     * Updates a norm for a collection of samples with a sample
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
     * generateAnomalyCellNorm
     * Description
     * @param cellNorm
     * @param images
     * @param preProcessingConfiguration
     * @param categoryName
     */
    void generateAnomalyCellNorm(std::array<float, 5> &cellNorm, std::map<std::string, cv::Mat> &images, const PreProcessing &preProcessingConfiguration, const std::string &categoryName) {


        int lastTotalAnomalyCells = 0;
        int totalAnomalyCells = 0;

        for (auto& [imageName, image] : images) {

            // Get image mask (to identify anomaly cells)
            std::string maskName = imageName.substr(0, imageName.size() - 3) + "png";
            cv::Mat imageMask = cv::imread(dataRoot + "masks/" + categoryName + "/" + maskName);

            if (imageMask.empty()) {
                std::cerr << "Warning: Could not load mask for " << imageName << "\n";
                continue;
            }

            // Apply pre-processing to image
            preProcessingConfiguration.apply(image);

            // Accumulate the distribution of pixel values accross anomaly cells
            for (int row = (cellSize / 2 ) + cellSize; row + cellSize < image.rows - cellSize; row += cellSize) {
                for (int col = (cellSize / 2) + cellSize; col + cellSize < image.cols - cellSize; col += cellSize) {

                    if (preProcessingConfiguration.lbp)
                        if (checkIfCellIsNormal(imageMask(cv::Range(row + 1, row + cellSize), cv::Range(col + 1, col + cellSize)))) 
                            continue; // skip normal cells
                    else
                        if (checkIfCellIsNormal(imageMask(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize)))) 
                            continue; // skip normal cells
                    totalAnomalyCells++;

                    // Get distribution of pixel values in anomaly cell
                    std::array<float, 5> LBPHistogram = {0};
                    cv::Mat cell = image(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
                    lbpValueDistribution(cell, LBPHistogram);

                    // Add to cumulative result
                    for (int j = 0; j < 5; j++)
                        cellNorm[j] += LBPHistogram[j];
                }
            }

            // Average the cummulated distribution for anomaly cells
            if (totalAnomalyCells == 0 || totalAnomalyCells <= lastTotalAnomalyCells) {
                std::cerr << "totalAnomalyCells == 0\n";
                return; // Need to investigate why this is happening if its still happening
            }
            lastTotalAnomalyCells = totalAnomalyCells;

        }

         // Average the cumulative distribution across all anomaly cells
        float sum = 0.0f;
        for (int k = 0; k < 5; k++) {
            cellNorm[k] /= totalAnomalyCells;
            cellNorm[k] /= images.size();
            sum += cellNorm[k];
        }
        
        // Normalize from 0 to 100
        if (sum > 0.0f) {
            for (int k = 0; k < 5; k++) {
                cellNorm[k] = (cellNorm[k] / sum) * 100.0f;
            }
        }
    }

    /*
     * generateNormalCellNorm
     * Description
     * @param cellNorm
     * @param images
     * @param preProcessingConfiguration
     */
    void generateNormalCellNorm(std::array<float, 5> &cellNorm, std::vector<cv::Mat> &images, const PreProcessing &preProcessingConfiguration) {

        for (auto& image : images) {

            // Apply pre-processing to image
            preProcessingConfiguration.apply(image);

            // Accumulate the distribution of pixel values accross anomaly cells
            for (int row = cellSize / 2; row + cellSize < image.rows; row += cellSize) {
                for (int col = cellSize / 2; col + cellSize < image.cols; col += cellSize) {

                    // Get distribution of pixel values in anomaly cell
                    std::array<float, 5> LBPHistogram = {0};
                    cv::Mat cell = image(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
                    lbpValueDistribution(cell, LBPHistogram);

                    // Add to cumulative result
                    for (int j = 0; j < 5; j++)
                        cellNorm[j] += LBPHistogram[j];
                }
            }
        }

         // Average the cumulative distribution across all images
        float sum = 0.0f;
        for (int k = 0; k < 5; k++) {
            cellNorm[k] /= images.size();
            sum += cellNorm[k];
        }
        
        // Normalize from 0 to 100
        if (sum > 0.0f) {
            for (int k = 0; k < 5; k++) {
                cellNorm[k] = (cellNorm[k] / sum) * 100.0f;
            }
        }
        std::cout << "\n";
        for (int i = 0; i < 5; i++)
            std::cout << "cellNorm: " << cellNorm[i] << "\n";
    }
}
