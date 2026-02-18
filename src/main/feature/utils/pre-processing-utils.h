#ifndef pre_processing_utils_H
#define pre_processing_utils_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Fault Sense
#include "../../objects/HSV.h"
#include "../../objects/CannyThreshold.h"
#include "../../objects/PixelCoordinates.h"

void thresholdHSV(cv::Mat& image, HSV& threshold);
void edgeDetection(cv::Mat& image, cv::Mat& kernal, CannyThreshold& threshold);
void removeNoise(cv::Mat& img, int minGrpSize);
void illuminationInvariance(const cv::Mat &image, cv::Mat &returnImage);

#endif
