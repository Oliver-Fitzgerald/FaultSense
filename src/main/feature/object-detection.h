#ifndef OBJECTDETECTION_H
#define OBJECTDETECTION_H

#include "objects/PixelCoordinates.h"
void objectDetection(cv::Mat &inputImage, cv::Mat &returnImage);
objectCoordinates getObject(cv::Mat &img);

#endif
