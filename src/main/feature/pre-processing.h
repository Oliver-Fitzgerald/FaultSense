#ifndef PreProcessing_H
#define PreProcessing_H

#include "../objects/PixelCoordinates.h"

void markFaultLBP(std::vector<std::array<float, 5>>&normalSampe, std::vector<std::array<float, 5>>& anomolySample, cv::Mat &image, std::string& imageCategory, const cv::Mat& imageMask, ObjectCoordinates& objectBounds);
void illuminationInvariance(const cv::Mat &image, cv::Mat &returnImage);

#endif
