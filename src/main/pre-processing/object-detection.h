#ifndef object_detection_H
#define object_detection_H


// OpenCV2
#include <opencv2/opencv.hpp>
// Fault Sense
#include "../objects/ObjectCoordinates.h"

void objectDetection(cv::Mat &inputImage, cv::Mat &returnImage);
void objectDetection(cv::Mat &inputImage, cv::Mat &returnImage, ObjectCoordinates& objectBounds);
ObjectCoordinates getObject(cv::Mat &img);

#endif
