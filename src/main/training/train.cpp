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

void trainAnomaly(std::map<std::string, std::array<float, 5>> &anomaly);

std::string dataRoot = "../../../data/";
std::string objectCategories[12] = {
    "candle/",
    "capsules/",
    "cashew/",
    "chewinggum/",
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
void trainNormal(cv::Mat normal) {
}

/*
 * trainAnomaly
 *
 * @param anomaly A mapping of object type (string) to it's average anomaly distribution
 */
void trainAnomaly(std::map<std::string, std::array<float, 5>> &anomaly) {

    int cellSize = 60; // shoulde be divisible by 2

    for (int i = 0; i < 12; i++) {
        std::map<std::string, cv::Mat> images = readImagesFromDirectory(dataRoot + objectCategories[i] + anomalyPath); 
        std::array<float, 5> averageLBPDistribution = {};
        for (const auto& [imageName, image] : images) {

            // Get image mask
            std::string maskName = imageName;
            maskName.replace(imageName.size() - 3, static_cast<std::string::size_type>(3), std::basic_string("png"));
            cv::Mat imageMask = cv::imread(dataRoot + "masks/" + objectCategories[i] + maskName);
            //re-size image mask to match LBPValues dimensions

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
                    for (int i = 0; i < 5; i++)
                        averageLBPDistribution[i] = LBPHistogram[i];
                }
            }

            // Average across all cells in the image
            for (int i = 0; i < averageLBPDistribution.size(); i++)
                averageLBPDistribution[i] = (averageLBPDistribution[i] / anomalyCells);

        }

        // Average across all images and record result
        for (int i = 0; i < averageLBPDistribution.size(); i++)
            averageLBPDistribution[i] = (averageLBPDistribution[i] / images.size()) * 100;


        std::string objectCategory = objectCategories[i];
        objectCategory.pop_back();
        anomaly.insert({objectCategory, averageLBPDistribution});
    }
}
