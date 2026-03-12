#ifndef pre_processing_H
#define pre_processing_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Fault Sense
#include "../objects/HSV.h"
#include "../objects/CannyThreshold.h"

void lbpValues(const cv::Mat& image, cv::Mat& LBPValues);
void thresholdHSV(cv::Mat& image, HSV& threshold);
void edgeDetection(cv::Mat& image, cv::Mat& kernal, CannyThreshold& threshold);
void removeNoise(cv::Mat& image, int minGrpSize);

#endif
