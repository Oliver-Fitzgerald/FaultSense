/*
 * common
 */

// Standard
#include <iostream>
#include <string>
#include <vector>
#include <exception>
// OpenCV2
#include <opencv2/opencv.hpp>
// Fault Sense
#include "feature/objects/CannyThreshold.h"
#include "feature/utils/pre-processing-utils.h"
#include "common.h"

bool classify(const float& normalDistance, const float& anomalyDistance, const std::string& category, const int index) {

    if (category == "chewinggum") {
        float normalDistanceAverage = 6;
        float anomalyDistanceAverage = 24;

        float localNormalDistance = std::abs(normalDistanceAverage - normalDistance);
        float localAnomalyDistance = std::abs(anomalyDistanceAverage - anomalyDistance);

        if (localNormalDistance < localAnomalyDistance) return true; 
        return false;

    } else if (category == "cashew") {

        if (index == 0) {
            float normalDistanceAverage = 0.1;
            float anomalyDistanceAverage = 42;

            float localNormalDistance = std::abs(normalDistanceAverage - normalDistance);
            float localAnomalyDistance = std::abs(anomalyDistanceAverage - anomalyDistance);

            if (localNormalDistance < localAnomalyDistance) return true; 
            return false;
        } else if (index == 1) {
            float normalDistanceAverage = 15;
            float anomalyDistanceAverage = 1;

            float localNormalDistance = std::abs(normalDistanceAverage - normalDistance);
            float localAnomalyDistance = std::abs(anomalyDistanceAverage - anomalyDistance);

            if (localNormalDistance < localAnomalyDistance) return false; 
            return true;
        } else
            throw std::runtime_error("Classification feature paramaters not initalized for categories(" + category + ")  for feature(" + std::to_string(index) + ")");

    } else
        throw std::runtime_error("Classification features not initalized for " + category);
}

void applyPreProcessing(cv::Mat& cell, const std::string& imageCategory, const int index) {

    if (imageCategory == "chewinggum") {
        CannyThreshold threshold{57, 29};
        cv::Mat kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
        edgeDetection(cell, kernal, threshold);

    } else if (imageCategory == "cashew") {

        if (index == 0) {
            HSV HSVThreshold{10, 13, 207, 255, 156,255};
            thresholdHSV(cell, HSVThreshold);
        } else if (index == 1) {
            HSV HSVThreshold{0, 179, 0, 255, 0,158};
            thresholdHSV(cell, HSVThreshold);
            removeBusyNoise(cell, 997);
            removeNoise(cell, 864);

        } else
            throw std::runtime_error("Pre-processing steps out of bounds check categoryFeatureCount in fault-sense-cli.cpp");

    } else
        throw std::runtime_error("Pre-processing not initalized in markFaultsLBP for " + imageCategory);
}

bool isNormal(cv::Mat& cell) {

    for (int row = 0; row < cell.rows; row++) {
        for (int col = 0; col < cell.cols; col++) {
            int pixel = cell.at<uchar>(row,col);

            if (pixel == 255) return false;
            else if (pixel != 0) throw std::invalid_argument("Cell mask contains a value that is not 0 or 255");
        }
    }

    return true;
}

int closestCommonFactor(int a, int b, int target = 60) {
    std::vector<int> commonFactors;

    // Find GCD first — common factors are factors of the GCD
    int g = std::__gcd(a, b);

    for (int i = 1; i * i <= g; i++) {
        if (g % i == 0) {
            commonFactors.push_back(i);
            if (i != g / i)
                commonFactors.push_back(g / i);
        }
    }

    // Find the factor closest to target
    return *std::min_element(commonFactors.begin(), commonFactors.end(),
        [target](int x, int y) {
            return abs(x - target) < abs(y - target);
        });
}

void binaryMaskExtraction(cv::Mat& img16, cv::Mat& img8) {

     if (img16.empty()) {
        throw std::runtime_error("Failed to load image");
    }

    // Ensure it's 16-bit 3-channel
    if (img16.type() != CV_8UC3) {
        throw std::runtime_error("Unexpected image type: " + std::to_string(img16.type()) + ", channels: " + std::to_string(img16.type()));
    }

    // 1. Extract one channel (B, G, or R — they are equal)
    cv::Mat channel16;
    cv::extractChannel(img16, channel16, 0);

    // 2. Convert to 8-bit (scale 0–65535 → 0–255)
    cv::Mat channel8;
    channel16.convertTo(channel8, CV_8U, 1.0 / 256.0);

    // 3. Threshold to binary (0 or 255)
    cv::threshold(channel8, img8, 0, 255, cv::THRESH_BINARY);
}

// USE THIS FOR CHECKING PRE-PROCSSING CONFIGURATION
/* cv::Mat original = cell.clone();
cv::Mat temp = adaptive_exposure(cell, 3.0, 1.0);
//binaryMaskExtraction(temp, cell);

CannyThreshold threshold{26, 14};
cv::Mat kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
edgeDetection(cell, kernal, threshold);
*/
/* I THINK TO MANY FN
 int number = 0; // global scope
HSV HSVThreshold{12, 150, 42, 100, 189,255};
thresholdHSV(cell, HSVThreshold);
removeNoise(cell, 200);
if (number > 500)
while (cv::pollKey() != 113) {
    cv::imshow("[]", cell);
    cv::imshow("44", original);
}
number++;

*/
