/*
 * features
 * This file contains functions for extracting features from images
 */

// OpenCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Standard
#include <iostream>
// Fault Sense
#include "HSV.h"
#include "features.h"

void markFault(cv::Mat& image, int minX, int maxX, int minY, int maxY, const char* label);
void thresholdHSV(cv::Mat& image, HSV& threshold);
void edgeDetection(cv::Mat& image, cv::Mat& kernal);


/*
 * markFault
 * given each edge point of a fault it draws a square to contain the fault and a label
 * to tag the square with
 */
void markFault(cv::Mat& image, int minX, int maxX, int minY, int maxY, const char* label) {

    cv::rectangle(image, cv::Point(minX - 10, minY - 10),cv::Point(maxX + 10, maxY + 10),cv::Scalar(0,0,255),3);
    putText(image, label, cv::Point(minX - 20, minY - 20), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,0,255), 2);
}

/*
 * thresholdHSV
 * Applys a given color threshold to an image to highligth regions of an image
 */
void thresholdHSV(cv::Mat& image, HSV& threshold) {

    cv::cvtColor(image, image, cv::COLOR_BGR2HSV);
    cv::Scalar lower(threshold.hueMin, threshold.saturationMin, threshold.valueMin);
    cv::Scalar upper(threshold.hueMax, threshold.saturationMax, threshold.valueMax);
    cv::inRange(image, lower, upper, image);
}

/*
 * edgeDetection
 * applys edget detection an image and erodes the image with a given kernal
 */
void edgeDetection(cv::Mat& image, cv::Mat& kernal) {

    cv::Canny(image, image, 100, 200);
    cv::erode(image, image, kernal);
}

/*
 * main
 * for testing functionality
int main(int argc, char **argv) {

    std::string testImage = "../../../data/sample-images/board-scratch.JPG";
    cv::Mat image = cv::imread(testImage);
    cv::Mat markFaultImg = image, thresholdHSVImg = image, edgeDetection = image;

    cv::Mat kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
    markFault(markFaultImg, 400, 500, 330, 600, "Scratch");
    cv::imshow("Mark Fault", markFaultImg);

    HSV threshold{79, 179, 9, 52,10,255};
    thresholdHSV(thresholdHSVImg, threshold);
    cv::imshow("Mark Fault", thresholdHSVImg);

    edgeDetection(cv::Mat& image, cv::Mat& kernal);

    bool next = true;
    while (next) {

        int keyPressed = cv::pollKey();
        if (keyPressed == 'q')
            next = false;
    }
}
 */
