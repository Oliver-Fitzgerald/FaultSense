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

cv::Mat LBPValues; // Ignore this not important

void illuminationInvariance(cv::Mat &image, cv::Mat &returnImage);
void checkFaultLBP(int (&normalSampe)[5], int (&anomolySample)[5], cv::Mat &image);


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
    checkFaultLBP(normalSampe, anomolySample,image);
}
*/

void illuminationInvariance(cv::Mat &image, cv::Mat &returnImage) {

    // Applying illumination invariance
    cv::Mat temp;
    cv::cvtColor(image, temp, cv::COLOR_BGR2GRAY);
    returnImage = brigthen_darker_areas(temp, 169, 46);
}


void checkFaultLBP(int (&normalSampe)[5], int (&anomolySample)[5], cv::Mat &image) {
    cv::Mat LBPValues;// IGNORRE THIS, to be deleted

    int cellSize = 26;
    if (cellSize % 2 != 0) throw std::invalid_argument("cellSize must be a multiple of 2");

    // Group Cells to from histogram
    for (int row = cellSize / 2; row < image.rows - cellSize; row += cellSize) {
        for (int col = cellSize / 2; col < image.cols - cellSize; col += cellSize) {

            // Get Cell
            cv::Rect cellDimensions = cv::Rect(col,row, cellSize, cellSize);
            cv::Mat cell = image(cellDimensions);

            // Compute LBP Histogram
            int cellLBPHistogram[5] = {0};
            computeLBP(cell, LBPValues, cellLBPHistogram);

            // Compare with normal and anomoly
            
            
            // Mark normal or anomoly
            markFault(image, col, col + cellSize, row , row + cellSize);
        }
    }

    cv::imshow("Image", image);
    while (true) cv::pollKey();
}
