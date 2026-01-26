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

    for (int i = 0; i < 12; i++) {
        std::map<std::string, cv::Mat> images = readImagesFromDirectory(dataRoot + objectCategories[i] + anomalyPath); 
        std::array<float, 5> averageLBPDistribution = {};
        for (const auto& [imageName, image] : images) {

            // Determine LBP distribution for image
            cv::Mat LBPValues; std::array<float, 5> LBPHistogram = {};
            lbpValues(image, LBPValues);
            lbpValueDistribution(LBPValues, LBPHistogram);

            // Add to cummulitive result of all samples
            for (int i = 0; i < 5; i++)
                averageLBPDistribution[i] = LBPHistogram[i];
        }

        // Average across all samples and record result
        for (int i = 0; i < averageLBPDistribution.size(); i++)
            averageLBPDistribution[i] = (averageLBPDistribution[i] / images.size()) * 100;

        std::string objectCategory = objectCategories[i];
        objectCategory.pop_back();
        std::cout << "Average " << objectCategory << " LBP Distribution: " << averageLBPDistribution[0] << ", " << averageLBPDistribution[1] << ", " << averageLBPDistribution[2] << ", " << averageLBPDistribution[3] << ", " << averageLBPDistribution[4] << "\n\n";
        anomaly.insert({objectCategory, averageLBPDistribution});
    }
}
