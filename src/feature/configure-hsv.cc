// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Standard
#include <iostream>
// Fault Sense
#include "HSV.h"


int hueMin = 0, hueMax = 179;
int saturationMin = 0, saturationMax = 255;
int valueMin = 0, valueMax = 255;

int main(int argc, char** argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    if (argc != 2) {
        std::cerr << "Invalid Usage\nCorrect Usage: " << *argv << " <image_path>" << std::endl;
        return -1;
    }

    argv++;
    std::string imagePath = *argv;
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr << "Error: Invalid image path: " << imagePath << std::endl;
        return -1;
    }

    cv::Mat imageHSV, mask;
    cv::cvtColor(image, imageHSV, cv::COLOR_BGR2HSV);

    cv::namedWindow("Trackbars", (640,200));
    cv::createTrackbar("Hue Min", "Trackbars", &hueMin, 179);
    cv::createTrackbar("Hue Max", "Trackbars", &hueMax, 179);
    cv::createTrackbar("Saturation Min", "Trackbars", &saturationMin, 255);
    cv::createTrackbar("Saturation Max", "Trackbars", &saturationMax, 255);
    cv::createTrackbar("Value Min", "Trackbars", &valueMin, 255);
    cv::createTrackbar("Value Max", "Trackbars", &valueMax, 255);

    bool next = true;
    while (next) {

        cv::Scalar lower(hueMin, saturationMin, valueMin);
        cv::Scalar upper(hueMax, saturationMax, valueMax);

        cv::inRange(imageHSV, lower, upper, mask);

        cv::imshow("HSV Image", mask);
        cv::waitKey(1);
    }
}
