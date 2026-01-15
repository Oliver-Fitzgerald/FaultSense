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
// Fault Sense
#include "utils/beta/texture-beta.h"
#include "utils/generic-utils.h"
#include "utils/pre-processing-utils.h"
#include "objects/RGB.h"
// Standard
#include <array>


void markFaultLBP(const std::array<float, 5>& normalSample, const std::array<float, 5>& anomolySample, cv::Mat &image);

/*
 * markFaultLBP
 */
void markFaultLBP(const std::array<float, 5>& normalSample, const std::array<float, 5>& anomolySample, cv::Mat &image) {
    cv::Mat LBPValues;// IGNORRE THIS, to be deleted
    int cellSize = 60;

    if (std::size(normalSample) != std::size(anomolySample)) throw std::invalid_argument("normalSample and anomolySample size must be equal");
    if (cellSize % 2 != 0) throw std::invalid_argument("cellSize must be a multiple of 2");

    // Group Cells to from histogram
    for (int row = cellSize / 2; row < image.rows - cellSize; row += cellSize) {
        for (int col = cellSize / 2; col < image.cols - cellSize; col += cellSize) {

            // Get cell
            cv::Rect cellDimensions = cv::Rect(col,row, cellSize, cellSize);
            cv::Mat cell; cv::Mat cellRaw = image(cellDimensions);
            illuminationInvariance(cellRaw, cell);

            // Compute LBP histogram for cell
            std::array<float, 5> cellLBPHistogram = {0};
            lbpValues(cellRaw, LBPValues);
            lbpValueDistribution(LBPValues, cellLBPHistogram);

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
