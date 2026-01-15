/*
 * texture
 * Contains functions for analysis the texture of an image
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
#include "../pre-processing-utils.h"
#include "../generic-utils.h"
#include "../../object-detection.h"

void lbpValueDistribution(const cv::Mat &LBPValues, std::array<float, 5>& LBPHistogram);
void lbpValues(const cv::Mat &image, cv::Mat &LBPValues);
uint8_t pixelLBP(const cv::Mat &image, const int x, const int y);

/*
int amount = 0;
int threshold = 0;
int noise = 0;

int main(int argc, char** argv) {

    cv::Mat originalImage = cv::imread("../../../../../../data/chewinggum/Data/Images/Anomaly/004.JPG");
    cv::Mat rawImage = cv::imread("../../../../../../data/chewinggum/Data/Images/Anomaly/004.JPG");
    //cv::Mat originalimage = cv::imread("../../../../../../data/sample-images/chewinggum-anomoly.jpg");
   // cv::Mat rawimage = cv::imread("../../../../../../data/sample-images/chewinggum-anomoly.jpg");
    cv::Mat LBPImage, croppedImage;

    // object detection
    HSV HSVThreshold{0, 22, 0, 119, 88,255}; thresholdHSV(rawImage, HSVThreshold);
    removeNoise(rawImage, 2000);
    objectCoordinates objectBounds = getObject(rawImage);
    crop(originalImage, objectBounds.yMin, objectBounds.yMax, objectBounds.xMin, objectBounds.xMax, croppedImage);


    // Show image
    cv::imshow("Image", LBPValues);
    cv::imshow("Image1",LBPImage);
    cv::imshow("Image2",finalImage);
    while (true) cv::pollKey();

    cv::namedWindow("Trackbars", (640,200));
    cv::createTrackbar("threshold", "Trackbars", &threshold, 255);
    cv::createTrackbar("amount", "Trackbars", &amount, 255);
    cv::createTrackbar("noise", "Trackbars", &noise, 1000000);
    int prev_threshold = threshold + 1;
    int prev_amount = amount + 1;
    int prev_noise = 0;
    cv::Mat exposed_img;

    cv::Mat temp, temp1;
    int paddingx = (3 - (croppedImage.cols % 3));
    int paddingy = (3 - (croppedImage.rows % 3));
    cv::cvtColor(croppedImage, temp, cv::COLOR_BGR2GRAY);
    padImage(temp, paddingy, paddingx, temp1);
    LBPImage = brigthen_darker_areas(temp1, threshold, amount);

    // Compute LBP for each 3x3 Cells
    cv::Mat LBPValues;
    int LBPHistogramWholeImage[5] = {0};
    computeLBP(LBPImage, LBPValues, LBPHistogramWholeImage);


    // Display Images
    while (true) {
        // Only reprocess if trackbar values changed
        if (threshold != prev_threshold || amount != prev_amount || noise != prev_noise) {
            std::cout << "Update\n";
            std::cout << "threshold: (" << prev_threshold << ", " << threshold << ")\n";
            std::cout << "amount: (" << prev_amount << ", " << amount << ")\n";
            std::cout << "noise: (" << prev_noise << ", " << noise << ")\n\n";


            // crop image to bounds divisible by 3
            cv::Mat temp, temp1;
            int paddingx = (3 - (croppedImage.cols % 3));
            int paddingy = (3 - (croppedImage.rows % 3));
            cv::cvtColor(croppedImage, temp, cv::COLOR_BGR2GRAY);
            padImage(temp, paddingy, paddingx, temp1);
            LBPImage = brigthen_darker_areas(temp1, threshold, amount);

            // Compute LBP for each 3x3 Cells
            cv::Mat LBPValues;
            int[5] LBPHistogram
            computeLBP(LBPImage, LBPValues, LBPHistogram);

            prev_threshold = threshold;
            prev_amount = amount;
            cv::imshow("Image", LBPValues);
        }
        
        int key = cv::waitKey(30); // Use waitKey instead of pollKey for better CPU usage
        if (key == 27 || key == 'q') break; // ESC or 'q' to exit
    }

}
*/

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
