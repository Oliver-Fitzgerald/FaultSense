/*
 * This is a temporart file for testing, to be deleted!
 */

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Standard
#include <iostream>
#include <filesystem>
#include <bitset>
#include <cstdint>
#include <cmath>
// Fault Sense
#include "../../feature/pre-processing.h"
#include "../../feature/object-detection.h"
#include "../../feature/utils/beta/texture-beta.h"
#include "../../feature/utils/generic-utils.h"
#include "../../feature/utils/pre-processing-utils.h"
#include "../../feature/objects/PixelCoordinates.h"


int main(int argc, char** argv) {

    cv::Mat LBPValues; // This is not used, just ignore it for now
    cv::Mat originalImage = cv::imread("../../../../data/sample-images/chewinggum-anomoly.JPG");
    cv::Mat rawImage = originalImage;

    // Object detection
    cv::Mat croppedImage;
    HSV HSVThreshold{0, 22, 0, 119, 88,255}; thresholdHSV(rawImage, HSVThreshold);
    removeNoise(rawImage, 2000);
    objectCoordinates objectBounds = getObject(rawImage);
    crop(originalImage, objectBounds.yMin, objectBounds.yMax, objectBounds.xMin, objectBounds.xMax, croppedImage);

    // Prompt user to select normal region
    cv::Mat normal; std::array<float, 5> LBPHistogramNormal = {0};
    cv::Rect normalArea = cv::selectROI("Select ROI", croppedImage);
    cv::Mat normalRaw = croppedImage(normalArea);
    illuminationInvariance(normalRaw, normal);
    lbpValues(normal, LBPValues);
    lbpValueDistribution(LBPValues, LBPHistogramNormal);

    // Prompt user to select anomaly region
    cv::Mat anomoly; std::array<float, 5> LBPHistogramAnomoly = {0};
    cv::Rect anomolyArea = cv::selectROI("Select ROI", croppedImage);
    cv::Mat anomolyRaw = croppedImage(anomolyArea);
    illuminationInvariance(anomolyRaw, anomoly);
    lbpValues(anomoly, LBPValues);
    lbpValueDistribution(LBPValues, LBPHistogramAnomoly);

    // Split image into cells and check if each cell is closer to normal or anomaly
    markFaultLBP(LBPHistogramNormal, LBPHistogramAnomoly, croppedImage);
}
