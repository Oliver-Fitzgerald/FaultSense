/*
 * train
 * This file contains functions to facilitate the training of models for
 * anomaly detection.
 */

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Fault Sense
#include "../feature/feature-extraction.h"
#include "../feature/utils/generic-utils.h"
// Standard
#include <map>
#include <array>
#include <string>
#include <cstdlib>

void trainAnomaly(std::map<std::string, std::array<float, 5>> &anomalyNorm);
void trainNormal(std::map<std::string, cv::Mat> &normalNorm);

std::string dataRoot = "../../../data/";
std::string objectCategories[12] = {
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
std::string anomalyPath = "Data/Images/Anomaly";
std::string normalPath = "Data/Images/Normal";

/*
 * trainNormal
 *
 *
 */
void trainNormal(std::map<std::string, cv::Mat> &normalNorm) {

    int cellSize = 60; // shoulde be divisible by 2

    for (int i = 0; i < 12; i++) {

        std::cout << "Training norm for normal samples of " << objectCategories[i] << " objects\n";
        std::map<std::string, cv::Mat> images = readImagesFromDirectory(dataRoot + objectCategories[i] + anomalyPath); 

        std::map<std::string, cv::Mat>::iterator itterator = images.begin();
        cv::Mat sampleImage = itterator->second;

        cv::Mat tempImage;
        lbpValues(sampleImage, tempImage);
        int rowMargin = tempImage.rows % cellSize;
        int colMargin = tempImage.cols % cellSize;
        cv::Mat categoryNorm((tempImage.rows - rowMargin) / cellSize, (tempImage.cols - colMargin) / cellSize, CV_32FC(5));

        float total = 0;
        for (const auto& [imageName, image] : images) {

            // Get image mask
            std::string maskName = imageName;
            maskName.replace(imageName.size() - 3, static_cast<std::string::size_type>(3), std::basic_string("png"));
            cv::Mat imageMask = cv::imread(dataRoot + "masks/" + objectCategories[i] + maskName);

            // Get LBP values for each pixel
            cv::Mat LBPValues;
            lbpValues(image, LBPValues);

            int collIndex, rowIndex = 0; 
            for (int row = rowMargin / 2; row < LBPValues.rows - (rowMargin / 2); row += cellSize) {
                collIndex = 0;
                for (int col = colMargin / 2; col < LBPValues.cols - (colMargin / 2); col += cellSize) {

                    // Get LBP distribution of normal cells
                    cv::Mat cell = LBPValues(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
                    std::array<float, 5> LBPHistogram = {};
                    lbpValueDistribution(cell, LBPHistogram);

                    // Update the cell norm average acrossed all images
                    float total = 0;
                    float* cellDistribution = categoryNorm.ptr<float>(rowIndex,collIndex);// Update Mat
                    for (int j = 0; j < 5; j++) {

                        /*
                        if ((j == 0) && (collIndex <= 2) && (rowIndex == 0))
                            std::cout << "Value( " << collIndex << "): " << LBPHistogram[j] << "\n";
                            */

                        cellDistribution[j] += LBPHistogram[j];
                        total += LBPHistogram[j];
                    }
                    /*
                    std::cout << " 100 / ((float(LBPValues.cols)) * (float(LBPValues.rows))) \n";
                    std::cout << "(float(LBPValues.rows)): " << (float(LBPValues.rows)) << "\n";
                    std::cout << "(float(LBPValues.cols)): " << (float(LBPValues.cols)) << "\n";
                    std::cout << "weigth: " << 100 / ((float(LBPValues.cols)) * (float(LBPValues.rows))) << "\n";
                    */
                    /*
                    if (total > 101 || total < 99 || ((rowIndex == 0) && (collIndex == 2)) ) {

                        std::cout << "\n(rowIndex, collIndex): (" << rowIndex << ", " << collIndex << ")\n";
                        for (int j = 0; j < 5; j++) {
                            std::cout << "lBPHistogram[" << j << "]: " << LBPHistogram[j] << "\n";
                        }
                        for (int j = 0; j < 5; j++) {
                            std::cout << "cellDistribution[" << j << "]: " << cellDistribution[j] << "\n";
                        }
                        std::cout << "total of cell: " << total << "\n";
                    }
                    */

                    collIndex++;
                }
                rowIndex++;
            }

            break;
        }

        for (int row = 0; row < categoryNorm.rows; row++) {
            for (int col = 0; col < categoryNorm.cols; col++) {

                /*
                if (col == 2)
                    std::cout << "\n";
                    */

                // left off here
                float total = 0;
                float* cellDistribution = categoryNorm.ptr<float>(row,col);
                for (int j = 0; j < 5; j++) {
                    //cellDistribution[j] = images.size() / cellDistribution[j];
                    cellDistribution[j] = (cellDistribution[j] / images.size()) * 100;
                    //cellDistribution[j] = images.size() / cellDistribution[j];
                    /*
                    if (col == 2) {
                        std::cout << "cellDistribution[j].rawValue: " << cellDistribution[j] << "\n";
                    }*/

                    total += cellDistribution[j];
                }
                if (total > 101 || total < 99) 
                //if (row == 0 && col == 2) {
                    std::cout << "(row, col): (" << row << ", " << col << ")\n";
                    for (int j = 0; j < 5; j++)
                        std::cout << "cellDistribution[j]: " << cellDistribution[j] << "\n";
                    std::cout << "total of accumulated cell: " << total << "\n";
                }

            }
        }

        //std::cout << "total: " << total << "\n";

        std::string objectCategory = objectCategories[i];
        objectCategory.pop_back(); // remove '/'
        normalNorm.insert({objectCategory, categoryNorm});
        return;//return after chewinggum for testing
    }
}

/*
 * trainAnomaly
 *
 * @param anomaly A mapping of object type (string) to it's average anomaly distribution
 */
void trainAnomaly(std::map<std::string, std::array<float, 5>> &anomalyNorm) {

    int cellSize = 60; // shoulde be divisible by 2

    for (int i = 0; i < 12; i++) {
        std::map<std::string, cv::Mat> images = readImagesFromDirectory(dataRoot + objectCategories[i] + anomalyPath); 
        std::array<float, 5> averageLBPDistribution = {};
        for (const auto& [imageName, image] : images) {

            // Get image mask
            std::string maskName = imageName;
            maskName.replace(imageName.size() - 3, static_cast<std::string::size_type>(3), std::basic_string("png"));
            cv::Mat imageMask = cv::imread(dataRoot + "masks/" + objectCategories[i] + maskName);

            // Get LBP values for anomaly cells
            cv::Mat LBPValues; std::array<float, 5> LBPHistogram = {};
            lbpValues(image, LBPValues);

            int anomalyCells = 0;
            for (int row = cellSize / 2; row + cellSize < LBPValues.rows; row += cellSize) {
                for (int col = cellSize / 2; col + cellSize < LBPValues.cols; col += cellSize) {

                    if (checkIfCellIsNormal(imageMask(cv::Range(row + 1, row + cellSize), cv::Range(col + 1, col + cellSize)))) // Note we (+ 1) as LBPValues loses a pixel on each edge as part of it's construction
                        continue; // skip normal cells
                    anomalyCells++;

                    // Get LBP distribution of anomaly cells only
                    cv::Mat cell = LBPValues(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
                    lbpValueDistribution(cell, LBPHistogram);

                    // Add to cummulitive result of all samples
                    for (int j = 0; j < 5; j++)
                        averageLBPDistribution[j] = LBPHistogram[j];
                }
            }

            // Average across all cells in the image
            for (int k = 0; k < averageLBPDistribution.size(); k++)
                averageLBPDistribution[k] = (averageLBPDistribution[k] / anomalyCells);

        }

        // Average across all images and record result
        for (int l = 0; l < averageLBPDistribution.size(); l++)
            averageLBPDistribution[l] = (averageLBPDistribution[l] / images.size()) * 100;


        std::string objectCategory = objectCategories[i];
        objectCategory.pop_back();
        anomalyNorm.insert({objectCategory, averageLBPDistribution});
    }
}
