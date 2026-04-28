#ifndef OBJECTDETECTION_H
#define OBJECTDETECTION_H

#include "../objects/PixelCoordinates.h"

cv::Mat objectDetection(cv::Mat &inputImage, cv::Mat &returnImage, const std::string& imageCategory, ObjectCoordinates& objectBounds);

#endif
