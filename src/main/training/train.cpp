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
#include "../feature/object-detection.h"
#include "../feature/utils/pre-processing-utils.h"
#include "../feature/utils/generic-utils.h"
#include "../feature/objects/CannyThreshold.h"
#include "../feature/objects/CannyThreshold.h"
#include "../common.h"
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
int number = 0;
void trainCell(std::vector<std::array<float, 5>> &cellNorm, const bool normal,const std::string& category = "") {

    int cellSize = 60; // should be divisible by 2

    for (int index = 0; index < cellNorm.size(); index++) {
        std::map<std::string, cv::Mat> images;
        if (normal)
            images = readImagesFromDirectory(dataRoot + category + "/" + normalPath + "/"); 
        else
            images = readImagesFromDirectory(dataRoot + category + "/" + anomalyPath + "/"); 

        for (auto& [imageName, image] : images) {

            // Object Detection
            ObjectCoordinates objectBounds;
            cv::Mat output, input = image.clone();
            objectDetection(input, output, category, objectBounds);
            cv::Mat inputTemp = image.clone();
            crop(inputTemp, objectBounds.xMin, objectBounds.xMax, objectBounds.yMin, objectBounds.yMax, image);

            // Get image mask
            cv::Mat imageMask;
            if (!normal) {

                std::string maskName = imageName;
                maskName.replace(imageName.size() - 3, static_cast<std::string::size_type>(3), std::basic_string("png"));
                cv::Mat mask = cv::imread(dataRoot + "masks/" + category + "/" + maskName);
                crop(mask, objectBounds.xMin, objectBounds.xMax, objectBounds.yMin, objectBounds.yMax, imageMask);
            }

            number++;
            // Apply pre-processing to image
            applyPreProcessing(image, category, index);
            if (index == 1 &&  number > 500) {

                while (cv::pollKey() != 113)
                    cv::imshow("img", image);
            }

            int anomalyCells = 0;
            for (int row = cellSize / 2; row + cellSize < image.rows; row += cellSize) {
                for (int col = cellSize / 2; col + cellSize < image.cols; col += cellSize) {

                    if (!normal && checkIfCellIsNormal(imageMask(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize))))
                        continue; // skip normal cells
                    anomalyCells++;

                    // Get LBP distribution of anomaly cells only
                    std::array<float, 5> featureHistogram = {0};
                    cv::Mat cell = image(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
                    lbpValueDistribution(cell, featureHistogram);

                    // Add to cummulitive result of all samples
                    float total = 0;
                    std::array<float, 5> temp = {0};
                    for (int j = 0; j < 5; j++) {
                        total += featureHistogram[j];
                        temp[j] = featureHistogram[j];
                        cellNorm[index][j] += featureHistogram[j];
                    }
                    if (std::isnan(total) || std::isinf(total)) {
                        std::cerr << "nan error\n";
                        for (int f = 0; f < 5; f++)
                            std::cerr << "temp[ " << f << "]: " << temp[f] << "\n";
                        throw std::runtime_error("nan error featureHistogram");
                    }
                }
            }

            if (anomalyCells == 0) {
                std::cerr << "WARNING: anomallyCells == 0\n";
                continue; // Need to investigate why this is happening
            }
            // DEBUGING: if (anomalyCells == 0) for (int f = 0; f < 5; f++) std::cout << "cellNorm[index][" << f << "]: " << cellNorm[index][f] << "\n";

            // Average across all cells in the image
            float total = 0;
            std::array<float, 5> temp = {0};
            std::array<float, 5> before = {0};
            for (int k = 0; k < 5; k++) {
                before[k] = cellNorm[index][k];
                cellNorm[index][k] = cellNorm[index][k] / (anomalyCells);
                total += cellNorm[index][k];
                temp[k] = cellNorm[index][k];
            }
            if (std::isnan(total) || std::isinf(total)) {
                std::cerr << "nan error\n";
                std::cerr << "anomalyCells: " << anomalyCells << "\n"; 
                for (int f = 0; f < 5; f++)
                    std::cerr << "temp[" << f << "]: " << temp[f] << "\n";
                for (int f = 0; f < 5; f++)
                    std::cerr << "before[" << f << "]: " << before[f] << "\n";

                throw std::runtime_error("nan error cellNorm[index]");
            }

            // Nomralize distribution values to 0 - 100
            float sum = total;
            float total2 = 0;
            std::array<float, 5> temp2 = {0};
            for (int k = 0; k < 5; k++) {
                cellNorm[index][k] = (cellNorm[index][k] / sum) * 100.0f;
                total2 += cellNorm[index][k];
                temp2[k] = cellNorm[index][k];
            }
            if (std::isnan(total2) || std::isinf(total2)) {
                std::cerr << "nan error normalizeing distribution\n"; 

                std::cerr << "sum: " << sum << "\n";
                std::cerr << "total: " << total << "\n";
                for (int f = 0; f < 5; f++)
                    std::cerr << "temp[" << f << "]: " << temp[f] << "\n";
                for (int f = 0; f < 5; f++)
                    std::cerr << "temp2[" << f << "]: " << temp2[f] << "\n";

                throw std::runtime_error("nan error");
            }
        }

        // Average across all images and record result
        float total = 100; /* for (int index = 0; index < cellNorm[index].size(); index++) { cellNorm[index][index] = (cellNorm[index][index] / images.size()) * 100; total += cellNorm[index][index]; } */

        if (total > 101 || total < 99) {
            std::cerr << "cellNorm[index].size(): " << cellNorm[index].size() << "\n";
            std::cerr << "cellNorm[index][0]: " <<cellNorm[index][0] << "\n"; 
            std::cerr << "cellNorm[index][1]: " <<cellNorm[index][1] << "\n"; 
            std::cerr << "cellNorm[index][2]: " <<cellNorm[index][2] << "\n"; 
            std::cerr << "cellNorm[index][3]: " <<cellNorm[index][3] << "\n"; 
            std::cerr << "cellNorm[index][4]: " <<cellNorm[index][4] << "\n"; 
            throw std::runtime_error("Distribution accross images not normalized to 100: " + std::to_string(total));
        }
    }
}
