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
#include "../evaluation/evaluation.h"
#include "../training/train.h"
#include "../general/file-operations/training-data.h"
#include "../general/generic-utils.h"
#include "../objects/RGB.h"
#include "utils/beta/texture-beta.h"
#include "utils/pre-processing-utils.h"
#include "object-detection.h"
#include "feature-extraction.h"
#include "pre-processing.h"
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

            cv::Mat cell = image(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
            std::array<float, 5> cellLBPHistogram = {};

            CannyThreshold threshold{57, 29};
            cv::Mat kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
            edgeDetection(cell, kernal, threshold);

            lbpValueDistribution(cell, cellLBPHistogram);

            // Compare with normal and anomoly samples
            float normalDistance = 0; float anomolyDistance = 0;
            for (int i = 0; i < std::size(normalSample); i++) {
                normalDistance += std::abs(cellLBPHistogram[i] - normalSample[i]);
                //std::cout << "- normalSample[" << i << "]: " << normalSample[i] << "\n";
                //std::cout << "- anomalySample[" << i << "]: " << anomolySample[i] << "\n";
                anomolyDistance += std::abs(cellLBPHistogram[i] - anomolySample[i]);
            }

            // Mark anomoly
            if (anomolyDistance < normalDistance) {
                //std::cout << "\nanomaly\n";
                RGB colour = RGB{0,0,255};
                markFault(image, col, col + cellSize, row , row + cellSize, nullptr, colour);
            } else;
                //std::cout << "\nnormal\n";
            //std::cout << "anomalyDistance : " << anomolyDistance << "\nnormalDistance: " << normalDistance << "\n";
        }
    }

    /* Testing
    cv::imshow("Image", image);
    while (cv::pollKey() != 113);
    */
}

/*
 * markFaultLBP
 */
void markFaultLBP(cv::Mat normalSample, const std::array<float, 5>& anomolySample, cv::Mat &image) {
    cv::Mat LBPValues;// IGNORRE THIS, to be deleted
    int cellSize = 60;

    //if (std::size(normalSample) != std::size(anomolySample)) throw std::invalid_argument("normalSample and anomolySample size must be equal");
    if (cellSize % 2 != 0) throw std::invalid_argument("cellSize must be a multiple of 2");

    // Group Cells to from histogram
    int collIndex, rowIndex = 0; 
    for (int row = cellSize / 2; row < image.rows - cellSize; row += cellSize) {
        collIndex = 0;
        for (int col = cellSize / 2; col < image.cols - cellSize; col += cellSize) {

            cv::Mat cell = image(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
            std::array<float, 5> cellLBPHistogram = {};

            CannyThreshold threshold{57, 29};
            cv::Mat kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
            edgeDetection(cell, kernal, threshold);

            lbpValueDistribution(cell, cellLBPHistogram);

            float* normal = normalSample.ptr<float>(rowIndex,collIndex);

            // Compare with normal and anomoly samples
            float normalDistance = 0; float anomolyDistance = 0;
            for (int i = 0; i < 5; i++) {
                normalDistance += std::abs(cellLBPHistogram[i] - normal[i]);
                anomolyDistance += std::abs(cellLBPHistogram[i] - anomolySample[i]);
                std::cout << "- normalSample[" << i << "]: " << normal[i] << "\n";
                std::cout << "- anomalySample[" << i << "]: " << anomolySample[i] << "\n";
            }

            // Mark anomoly
            if (anomolyDistance < normalDistance) {
                std::cout << "\nanomaly\n";
                RGB colour = RGB{0,0,255};
                markFault(image, col, col + cellSize, row , row + cellSize, nullptr, colour);
            } else
                std::cout << "\nnormal\n";
            std::cout << "anomalyDistance : " << anomolyDistance << "\nnormalDistance: " << normalDistance << "\n";
            collIndex++;
        }
        rowIndex++;
    }

    /* Testing
    cv::imshow("Image", image);
    while (cv::pollKey() != 113);
    */
}
