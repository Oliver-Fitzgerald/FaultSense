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
            //HSV HSVThreshold{0, 22, 0, 119, 88,255}; thresholdHSV(cellRaw, HSVThreshold);
            illuminationInvariance(cellRaw, cell);

            
            //cv::Mat temp;
            //cv::bitwise_not(image, temp);
            //removeNoise(temp, 10);

            // Compute LBP Histogram
            float cellLBPHistogram[5] = {0};
            computeLBP(cell, LBPValues, cellLBPHistogram);

            //if (row == ((cellSize / 2) + (cellSize * 2)) && col == ((cellSize / 2) + (cellSize * 2))) {
                std::cout << "=======================\n";
                std::cout << "normalValues => (" << normalSample[0] << ", " << normalSample[1] << ", " << normalSample[2] << ", " << normalSample[3] << ", " << normalSample[4] << ")" << "\n";
                std::cout << "anomolyValues => (" << anomolySample[0] << ", " << anomolySample[1] << ", " << anomolySample[2] << ", " << anomolySample[3] << ", " << anomolySample[4] << ")" << "\n";
                std::cout << "cellValues => (" << cellLBPHistogram[0] << ", " << cellLBPHistogram[1] << ", " << cellLBPHistogram[2] << ", " << cellLBPHistogram[3] << ", " << cellLBPHistogram[4] << ")" << "\n";
                std::cout << "=======================\n";
            //}

            // Compare with normal and anomoly
            float normalDistance = 0; float anomolyDistance = 0;
            std::string normalDisStr = "("; std::string anomolyDisStr = "(";
            for (int i = 0; i < std::size(normalSample); i++) {
                normalDistance += std::abs(cellLBPHistogram[i] - normalSample[i]);
                normalDisStr += ", " + std::to_string(normalDistance);

                anomolyDistance += std::abs(cellLBPHistogram[i] - anomolySample[i]);
                anomolyDisStr += ", " + std::to_string(anomolyDistance);
            }

            // Mark normal or anomoly
            if (anomolyDistance < normalDistance) {
                RGB colour = RGB{0,0,255};
                markFault(image, col, col + cellSize, row , row + cellSize, nullptr, colour);
            }
            if (row == ((cellSize / 2) + (cellSize * 2)) && col == ((cellSize / 2) + (cellSize * 5))) {
                std::cout << "=======================\n";
                std::cout << "normalDistance => " << normalDistance << "\n";
                std::cout << "normalDistance => " << normalDisStr << "\n";
                std::cout << "anomolyDistance => " << anomolyDistance << "\n";
                std::cout << "anomolyDistance => " << anomolyDisStr << "\n";
                std::cout << "=======================\n";

            }
        }
    }

    // Testing
    cv::imshow("Image", image);
    while (true) cv::pollKey();
}
