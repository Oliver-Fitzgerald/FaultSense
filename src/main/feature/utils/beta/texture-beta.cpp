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
// Fault Sense
#include "../pre-processing-utils.h"
#include "../generic-utils.h"
#include "../../object-detection.h"

uint8_t computePixelLBP(cv::Mat &image, int x, int y);

int main(int argc, char** argv) {

    cv::Mat rawImage = cv::imread("../../../../../data/sample-images/chewinggum-anomoly.JPG");
    cv::Mat finalImage, LBPImage, croppedImage;

    // Object Detection
    HSV HSVThreshold{0, 22, 0, 119, 88,255}; thresholdHSV(rawImage, HSVThreshold);
    removeNoise(rawImage);
    objectCoordinates objectBounds = getObject(rawImage);
    crop(rawImage, objectBounds.yMin, objectBounds.yMax, objectBounds.xMin, objectBounds.xMax, croppedImage);

    // Crop Image to bounds divisible by 16
    if (croppedImage.rows % 16 != 0 || croppedImage.cols % 16 != 0)
        std::cout << "Image Needs to be padded\n";
    else
        std::cout << "Image has valid dimensions\n";
    int maxX = objectBounds.xMax + (objectBounds.xMax % 16);
    int maxY = objectBounds.yMax + (objectBounds.yMax % 16);
    padImage(croppedImage, maxX, maxY, LBPImage); // Temporarily Always cropping to correct dimensions
    /*
    
    // Compute LBP for each 16x16 Cell
    cv::Mat lbdValues = cv::Mat::zeros(LBPImage.rows / 16, LBPImage.cols / 16, CV_8U);
    for (int x = 1; x < LBPImage.cols - 1; x++) {
        for (int y = 1; y < LBPImage.cols - 1; y++) {
            uint8_t pixel = lbdValues.at<uint8_t>(x, y);
            pixel = computePixelLBP(LBPImage, x,y);

        }
    }

    finalImage = LBPImage;
    // Show image
    cv::imshow("Image", finalImage);
    while (true) cv::pollKey();
    */

}

uint8_t computePixelLBP(cv::Mat &image, int x, int y) {

    uint8_t LBDValue = 0b00000000;
    int centerValue = image.at<uchar>(x,y);
    // Above
    if (image.at<uchar>(x - 1, y - 1) >= centerValue) 
        LBDValue |= (1 << 0);
    if (image.at<uchar>(x - 1, y) >= centerValue) 
        LBDValue |= (1 << 0);
    if (image.at<uchar>(x - 1, y + 1) >= centerValue) 
        LBDValue |= (1 << 0);

    // Inline
    if (image.at<uchar>(x, y - 1) >= centerValue) 
        LBDValue |= (1 << 0);
    if (image.at<uchar>(x, y + 1) >= centerValue) 
        LBDValue |= (1 << 0);

    // Below
    if (image.at<uchar>(x + 1, y - 1) >= centerValue) 
        LBDValue |= (1 << 0);
    if (image.at<uchar>(x + 1, y) >= centerValue) 
        LBDValue |= (1 << 0);
    if (image.at<uchar>(x + 1, y + 1) >= centerValue) 
        LBDValue |= (1 << 0);
    
    return LBDValue;
}
