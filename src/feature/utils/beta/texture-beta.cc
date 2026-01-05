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

/*
 * texture
 * Contains functions for analysis the texture of an image
 */
int main(int argc, char** argv) {

    // Object Detection
    cv::Mat finalImage, fault, tmp, img, image = cv::imread("../data/sample-images/chewinggum-anomoly.JPG");
    HSV HSVThreshold{0, 22, 0, 119, 88,255};
    thresholdHSV(image, HSVThreshold);
    tmp = img;
    removeNoise(image);
    objectCoordinates coordinates = getObject(img);
    cv::Mat tmp1;
    crop(image, coordinates.yMin, coordinates.yMax, coordinates.xMin, coordinates.xMax, tmp1);
    cv::cvtColor(tmp1, fault, cv::COLOR_BGR2GRAY);

    // Divide window into 16x16 Cells
    if (fault.rows % 16 != 0 || fault.cols % 16 != 0)
        std::cout << "Image Needs to be padded\n";
    else
        std::cout << "Image has valid dimensions\n";
    cv::Mat next;
    crop(image, coordinates.yMin + ((fault.cols % 16) / 2) + 1, coordinates.yMax - ((fault.cols % 16) / 2), coordinates.xMin + ((fault.rows % 16) / 2) + 1, coordinates.xMax - ((fault.rows % 16) / 2), next);
    
    cv::Mat lbdValues = cv::Mat::zeros(next.rows / 16, next.cols / 16, CV_8U);
    for (int x = 1; x < next.cols - 1; x++) {
        for (int y = 1; y < next.cols - 1; y++) {
            uint8_t pixel = lbdValues.at<uint8_t>(x, y);
            pixel = computePixelLBP(next, x,y);

        }
    }

    finalImage = next;
    // Show image
    cv::imshow("Image", finalImage);
    while (true) cv::pollKey();

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
