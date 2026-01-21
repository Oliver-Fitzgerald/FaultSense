/*
 * feature-extraction
 * Extracts quantitative features from processed images
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
#include <array>
// Fault Sense
#include "utils/pre-processing-utils.h"
#include "utils/generic-utils.h"
#include "object-detection.h"

void lbpValueDistribution(const cv::Mat &LBPValues, std::array<float, 5>& LBPHistogram);
void lbpValues(const cv::Mat &image, cv::Mat &LBPValues);
uint8_t pixelLBP(const cv::Mat &image, const int x, const int y);


/*
 * lbpValueDistribution
 */
void lbpValueDistribution(const cv::Mat &LBPValues, std::array<float, 5>& LBPHistogram) {

    // Calculate weigth of each pixel to normalize histogram from 0 - 100
    float weigth = 100 / ((float(LBPValues.cols - 2)) * (float(LBPValues.rows - 2)));

    for (int row = 1; row < LBPValues.rows - 1; row++) {
        for (int col = 1; col < LBPValues.cols - 1; col++) {

            int value = LBPValues.at<uint8_t>(row, col);
            int index = std::clamp(value / 51, 0, 4);
            LBPHistogram[index] += weigth;
        }
    }
}

/*
 * lbpValues
 *
 * @param image The image for which Local Binary Pattern values are computed
 * @param LBPValues The image of Local Computed Binary values computed for 3x3 cells
 */
void lbpValues(const cv::Mat &image, cv::Mat &LBPValues) {

    //Compute for every pixel
    // Form histogram of cell values in image
    LBPValues = cv::Mat::zeros(image.rows - 2, image.cols - 2, CV_8UC1);
    float weigth = 100 / ((float(image.cols - 2)) * (float(image.rows - 2)));

    for (int row = 1; row < image.rows - 1; row++) {
        for (int col = 1; col < image.cols - 1; col++) {

            int value =  pixelLBP(image, row, col);
            LBPValues.at<uint8_t>(row - 1 ,col - 1) = value;
        }
    }

}

/*
 * pixelLBP
 */
uint8_t pixelLBP(const cv::Mat &image, const int x, const int y) {

    uint8_t LBDValue = 0b00000000;
    int centerValue = image.at<uchar>(x,y);
    
    if (image.at<uchar>(x - 1, y - 1) >= centerValue) 
        LBDValue |= (1 << 0);
    if (image.at<uchar>(x - 1, y) >= centerValue) 
        LBDValue |= (1 << 1);
    if (image.at<uchar>(x - 1, y + 1) >= centerValue) 
        LBDValue |= (1 << 2);

    // Inline
    if (image.at<uchar>(x, y - 1) >= centerValue) 
        LBDValue |= (1 << 3);
    if (image.at<uchar>(x, y + 1) >= centerValue) 
        LBDValue |= (1 << 4);

    // Below
    if (image.at<uchar>(x + 1, y - 1) >= centerValue) 
        LBDValue |= (1 << 5);
    if (image.at<uchar>(x + 1, y) >= centerValue) 
        LBDValue |= (1 << 6);
    if (image.at<uchar>(x + 1, y + 1) >= centerValue) 
        LBDValue |= (1 << 7);
    
    return LBDValue;
}
