/*
 * shadow-detection-removal
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
// Fault Sense
#include "../pre-processing-utils.h"
#include "../generic-utils.h"
#include "../../object-detection.h"

int cannyX = 0, cannyY = 0;
int randA = 0, randB = 0;
int threshold = 0;
int amount = 0;

cv::Mat adaptive_exposure(const cv::Mat& img, float strength = 2.0, float curve = 2.0);
cv::Mat adaptive_exposure(const cv::Mat& img, float strength, float curve) {
    cv::Mat result = img.clone();
    
    for(int y = 0; y < img.rows; y++) {
        for(int x = 0; x < img.cols; x++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
            
            // Calculate intensity
            int intensity = (pixel[0] + pixel[1] + pixel[2]) / 3;
            float normalized = intensity / 255.0f;
            
            // Power curve: higher curve value = more weight to darks
            float boost = 1.0 + strength * pow(1.0 - normalized, curve);
            
            for(int c = 0; c < 3; c++) {
                result.at<cv::Vec3b>(y, x)[c] = 
                    cv::saturate_cast<uchar>(pixel[c] * boost);
            }
        }
    }
    return result;
}

cv::Mat brigthen_darker_areas(const cv::Mat& img, int threshold, int amount);
cv::Mat brigthen_darker_areas(const cv::Mat& img, int threshold, int amount) {

    cv::Mat returnImage = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {

            int pixel = img.at<uint8_t>(row, col);
            if (pixel < threshold)
                returnImage.at<uint8_t>(row,col) = pixel + amount;
            else
                returnImage.at<uint8_t>(row,col) = pixel;
        }
    }
    return returnImage;

}

int main (int argc, char** argv) {
    cv::Mat test = cv::imread("../../../../../../data/chewinggum/Data/Images/Anomaly/004.JPG");
    cv::Mat originalImage = cv::imread("../../../../../../data/chewinggum/Data/Images/Anomaly/004.JPG");
    cv::Mat rawImage = cv::imread("../../../../../../data/chewinggum/Data/Images/Anomaly/004.JPG");
    //cv::Mat originalImage = cv::imread("../../../../../../data/sample-images/chewinggum-anomoly.JPG");
   // cv::Mat rawImage = cv::imread("../../../../../../data/sample-images/chewinggum-anomoly.JPG");
    cv::Mat LBPImage, croppedImage;

    // Object Detection
    HSV HSVThreshold{0, 22, 0, 119, 88,255}; thresholdHSV(rawImage, HSVThreshold);
    removeNoise(rawImage, 2000);
    objectCoordinates objectBounds = getObject(rawImage);
    crop(originalImage, objectBounds.yMin, objectBounds.yMax, objectBounds.xMin, objectBounds.xMax, croppedImage);

    // Crop Image to bounds divisible by 3
    cv::Mat temp;
    int paddingX = (3 - (croppedImage.cols % 3));
    int paddingY = (3 - (croppedImage.rows % 3));
    padImage(croppedImage, paddingY, paddingX, temp);
    cv::cvtColor(temp, LBPImage, cv::COLOR_BGR2GRAY);


    cv::namedWindow("Trackbars", (640,200));
    cv::createTrackbar("Canny Threshold X", "Trackbars", &cannyX, 255);
    cv::createTrackbar("Canny Threshold Y", "Trackbars", &cannyY, 255);
    cv::createTrackbar("Random A", "Trackbars", &randA, 10);
    cv::createTrackbar("Random B", "Trackbars", &randB, 10);
    cv::createTrackbar("threshold", "Trackbars", &threshold, 255);
    cv::createTrackbar("amount", "Trackbars", &amount, 255);
    int prev_threshold = threshold;
    int prev_amount = amount;
    cv::Mat exposed_img;

    // 169, 46
    while (true) {
        // only reprocess if trackbar values changed
        if (threshold != prev_threshold || amount != prev_amount) {
            exposed_img = brigthen_darker_areas(LBPImage, threshold, amount);
            prev_threshold = threshold;
            prev_amount = amount;
            cv::imshow("image", exposed_img);
        }
        
        int key = cv::waitKey(30); // use waitkey instead of pollkey for better cpu usage
        if (key == 27 || key == 'q') break; // esc or 'q' to exit
    }
}

