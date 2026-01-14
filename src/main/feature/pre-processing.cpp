/*
 * pre-processing
 * Manages the pre-processing flows for object and feature extraction 
 * steps
 */

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//Fault Sense
#include "utils/beta/texture-beta.h"
#include "utils/generic-utils.h"
#include "utils/pre-processing-utils.h"
#include "objects/RGB.h"


void illuminationInvariance(cv::Mat &image, cv::Mat &returnImage);
void checkFaultLBP(float (&normalSample)[5], float (&anomolySample)[5], cv::Mat &image);


/*
int main(int argc, char** argv) {

    // Object detection
    HSV HSVThreshold{0, 22, 0, 119, 88,255}; thresholdHSV(rawImage, HSVThreshold);
    removeNoise(rawImage, 2000);
    objectCoordinates objectBounds = getObject(rawImage);
    crop(originalImage, objectBounds.yMin, objectBounds.yMax, objectBounds.xMin, objectBounds.xMax, croppedImage);

    // Prompt user to select normal region
    cv::Rect normalArea = cv::selectROI("Select ROI", croppedImage);
    cv::Mat normalRaw = img(normalArea);
    cv::Mat normal;
    illuminationInvariance(normalRaw, normal);
    int[5] LBPHistogramNormal;
    computeLBP(normal, LBPValues, LBPHistogramNormal);

    // Prompt user to select anomaly region
    cv::Rect anomolyArea = cv::selectROI("Select ROI", croppedImage);
    cv::Mat anomolyRaw = img(anomolyArea);
    cv::Mat anomoly;
    illuminationInvariance(anomolyRaw, anomoly);
    int[5] LBPHistogramAnomoly;
    computeLBP(anomoly, LBPValues, LBPHistogramAnomoly);

    // Split image into cells and check if each cell is closer to normal or anomaly
    checkFaultLBP(normalSample, anomolySample,image);
}
*/

void illuminationInvariance(cv::Mat &image, cv::Mat &returnImage) {

    // Applying illumination invariance
    cv::Mat temp;
    cv::cvtColor(image, temp, cv::COLOR_BGR2GRAY);
    returnImage = brigthen_darker_areas(temp, 169, 46);
}


void checkFaultLBP(float (&normalSample)[5], float (&anomolySample)[5], cv::Mat &image) {
    cv::Mat LBPValues;// IGNORRE THIS, to be deleted

    if (sizeof(normalSample) != sizeof(anomolySample)) throw std::invalid_argument("normalSample and anomolySample size must be equal");

    int cellSize = 60;
    if (cellSize % 2 != 0) throw std::invalid_argument("cellSize must be a multiple of 2");

    // Group Cells to from histogram
    int count = 0;
    for (int row = cellSize / 2; row < image.rows - cellSize; row += cellSize) {
        for (int col = cellSize / 2; col < image.cols - cellSize; col += cellSize) {
            count++;

            // Get Cell
            cv::Rect cellDimensions = cv::Rect(col,row, cellSize, cellSize);
            cv::Mat cell; cv::Mat cellRaw = image(cellDimensions);
            illuminationInvariance(cellRaw, cell);

            // Compute LBP histogram for cell
            float cellLBPHistogram[5] = {0};
            computeLBP(cell, LBPValues, cellLBPHistogram);

            // Compare with normal and anomoly samples
            float normalDistance = 0; float anomolyDistance = 0;
            for (int i = 0; i < std::size(normalSample); i++) {
                normalDistance += std::abs(cellLBPHistogram[i] - normalSample[i]);
                anomolyDistance += std::abs(cellLBPHistogram[i] - anomolySample[i]);
            }

            // Mark anomoly
            if (anomolyDistance < normalDistance) {
                RGB colour = RGB{0,0,255};
                markFault(image, col, col + cellSize, row , row + cellSize, nullptr, colour);
            }
        }
    }

    // Testing
    cv::imshow("Image", image);
    while (true) cv::pollKey();
}
