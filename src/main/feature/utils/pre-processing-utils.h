#pragma once
#include "../objects/HSV.h"
#include "../objects/CannyThreshold.h"
#include "../../objects/PixelCoordinates.h"
void thresholdHSV(cv::Mat& image, HSV& threshold);
void edgeDetection(cv::Mat& image, cv::Mat& kernal, CannyThreshold& threshold);
void removeNoise(cv::Mat& img, int minGrpSize);
void removeBusyNoise(cv::Mat& img, int maxGrpSize);
void illuminationInvariance(const cv::Mat &image, cv::Mat &returnImage);
cv::Mat adaptive_exposure(const cv::Mat& img, float strength = 2.0, float curve = 2.0);
