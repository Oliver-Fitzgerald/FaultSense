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
void trainCell(std::map<std::string, std::array<float, 5>> &cellNorm, const bool normal, const char* category);
void initNormMatrix(const std::map<std::string, cv::Mat>::iterator &itterator, int cellSize, cv::Mat &categoryNorm);
void updateCategoryNorm(cv::Mat norm, cv::Mat values, int cellSize, int numberOfSamples);

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

/*
 * trainMatrix
 */
void trainMatrix(std::map<std::string, cv::Mat> &matrixNorm) {

    int cellSize = 60; // shoulde be divisible by 2

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
 * THIS METHOD NEEDS TO BE TIDIED UP AND MAKE TRAINING TYPE CONFIGURABLE
 *
 * @param cellNorm A mapping of object type (string) to it's average anomaly distribution
 */
void trainCell(std::map<std::string, std::array<float, 5>> &cellNorm, const bool normal, const char* category = "") {

    int cellSize = 60; // shoulde be divisible by 2

    if (category[0] != '\0') {

        std::string imagePath;
        if (normal)
            imagePath = dataRoot + category + "/" + normalPath;
        else
            imagePath = dataRoot + category + "/" + anomalyPath;

        std::map<std::string, cv::Mat> images = readImagesFromDirectory(imagePath); 

        std::array<float, 5> averageLBPDistribution = {0};
        for (auto& [imageName, image] : images) {

            // Get image mask
            std::string maskName = imageName;
            maskName.replace(imageName.size() - 3, static_cast<std::string::size_type>(3), std::basic_string("png"));
            cv::Mat imageMask = cv::imread(dataRoot + "masks/" + category + "/" + maskName);

            // Get LBP values for anomaly cells
            cv::Mat LBPValues; 
            //lbpValues(image, LBPValues);

            CannyThreshold threshold{57, 29};
            cv::Mat kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
            edgeDetection(image, kernal, threshold);
            LBPValues = image.clone();

            int anomalyCells = 0;
            for (int row = cellSize / 2; row + cellSize < LBPValues.rows; row += cellSize) {
                for (int col = cellSize / 2; col + cellSize < LBPValues.cols; col += cellSize) {

                    if (checkIfCellIsNormal(imageMask(cv::Range(row + 1, row + cellSize), cv::Range(col + 1, col + cellSize)))) // Note we (+ 1) as LBPValues loses a pixel on each edge as part of it's construction
                        continue; // skip normal cells
                    anomalyCells++;

                    // Get LBP distribution of anomaly cells only
                    std::array<float, 5> LBPHistogram = {0};
                    cv::Mat cell = LBPValues(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
                    lbpValueDistribution(cell, LBPHistogram);

                    // Add to cummulitive result of all samples
                    for (int j = 0; j < 5; j++)
                        averageLBPDistribution[j] += LBPHistogram[j]; // Should this not be +=
                }
            }

            // Average across all cells in the image
            for (int k = 0; k < averageLBPDistribution.size(); k++)
                averageLBPDistribution[k] = (averageLBPDistribution[k] / anomalyCells);


            std::string objectCategory = category;
            cellNorm.insert({objectCategory, averageLBPDistribution});
            break;
        }

    } else {
        for (int i = 0; i < 12; i++) {

            std::map<std::string, cv::Mat> images;
            if (normal)
                images = readImagesFromDirectory(dataRoot + objectCategories[i] + normalPath); 
            else
                images = readImagesFromDirectory(dataRoot + objectCategories[i] + anomalyPath); 

            std::array<float, 5> averageLBPDistribution = {0};
            for (const auto& [imageName, image] : images) {

                // Get image mask
                std::string maskName = imageName;
                maskName.replace(imageName.size() - 3, static_cast<std::string::size_type>(3), std::basic_string("png"));
                cv::Mat imageMask = cv::imread(dataRoot + "masks/" + objectCategories[i] + maskName);

                // Get LBP values for anomaly cells
                cv::Mat LBPValues;
                lbpValues(image, LBPValues);

                int anomalyCells = 0;
                //std::cout << "new\n";
                for (int row = cellSize / 2; row + cellSize < LBPValues.rows; row += cellSize) {
                    for (int col = cellSize / 2; col + cellSize < LBPValues.cols; col += cellSize) {

                        if (checkIfCellIsNormal(imageMask(cv::Range(row + 1, row + cellSize), cv::Range(col + 1, col + cellSize)))) // Note we (+ 1) as LBPValues loses a pixel on each edge as part of it's construction
                            continue; // skip normal cells
                        anomalyCells++;
                        //std::cout << "anomalyCells update: "  << anomalyCells << "\n";

                        // Get LBP distribution of anomaly cells only
                        std::array<float, 5> LBPHistogram = {0};
                        cv::Mat cell = LBPValues(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
                        lbpValueDistribution(cell, LBPHistogram);

                        // Add to cummulitive result of all samples
                        float total = 0;
                        std::array<float, 5> temp = {0};
                        for (int j = 0; j < 5; j++) {
                            total +=LBPHistogram[j];
                            temp[j] = LBPHistogram[j];
                            averageLBPDistribution[j] += LBPHistogram[j];
                        }
                        if (std::isnan(total) || std::isinf(total)) {
                            std::cout << "nan error LBPHistogram\n";
                            for (int f = 0; f < 5; f++)
                                std::cout << "temp[ " << f << "]: " << temp[f] << "\n";
                            std::cout << "nan error\n";
                            std::cout << "objectCategory: " << objectCategories[i] << "\n";
                            return;
                        }
                    }
                }

                if (anomalyCells == 0) continue; // Need to investigate why this is happening
                if (anomalyCells == 0)
                    for (int f = 0; f < 5; f++)
                        std::cout << "averageLBPDistribution[" << f << "]: " << averageLBPDistribution[f] << "\n";
                float total = 0;
                // Average across all cells in the image
                std::array<float, 5> temp = {0};
                std::array<float, 5> before = {0};
                for (int k = 0; k < 5; k++) {
                    before[k] = averageLBPDistribution[k];
                    averageLBPDistribution[k] = averageLBPDistribution[k] / (anomalyCells);
                    total += averageLBPDistribution[k];
                    temp[k] = averageLBPDistribution[k];
                }
                if (std::isnan(total) || std::isinf(total)) {
                    std::cout << "nan error total\n"; 
                    std::cout << "anomalyCells: " << anomalyCells << "\n"; 
                    for (int f = 0; f < 5; f++)
                        std::cout << "temp[" << f << "]: " << temp[f] << "\n";
                    for (int f = 0; f < 5; f++)
                        std::cout << "before[" << f << "]: " << before[f] << "\n";
                    std::cout << "nan error\n";
                    std::cout << "objectCategory: " << objectCategories[i] << "\n";
                    return;
                }

                float sum = total;
                float total2 = 0;
                std::array<float, 5> temp2 = {0};
                for (int k = 0; k < 5; k++) {
                    averageLBPDistribution[k] = (averageLBPDistribution[k] / sum) * 100.0f;
                    total2 += averageLBPDistribution[k];
                    temp2[k] = averageLBPDistribution[k];
                }
                if (std::isnan(total2) || std::isinf(total2)) {
                    std::cout << "nan error total2\n"; 
                    std::cout << "sum: " << sum << "\n";
                    std::cout << "total: " << total << "\n";
                    for (int f = 0; f < 5; f++)
                        std::cout << "temp[" << f << "]: " << temp[f] << "\n";
                    for (int f = 0; f < 5; f++)
                        std::cout << "temp2[" << f << "]: " << temp2[f] << "\n";
                    std::cout << "nan error\n";
                    std::cout << "objectCategory: " << objectCategories[i] << "\n";
                    return;
                }
            }

            // Average across all images and record result
            float total = 0;
            for (int l = 0; l < averageLBPDistribution.size(); l++) {
                averageLBPDistribution[l] = (averageLBPDistribution[l] / images.size()) * 100;
                total += averageLBPDistribution[l];
            }
            if (total > 101 || total < 99) {
                std::cerr << "total not == 100: " << total << "\n";
                return;
            }


            std::string objectCategory = objectCategories[i];
            objectCategory.pop_back();
            cellNorm.insert({objectCategory, averageLBPDistribution});
        }
    }
}
