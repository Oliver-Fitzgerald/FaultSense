#ifndef METHODS_H
#define METHODS_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Fault Sense
#include "../objects/RGB.h"
// Standard
#include <string>
#include <map>


void markFault(cv::Mat& image, int minX, int maxX, int minY, int maxY, const char* label, RGB colour);
void crop(cv::Mat& image, int minX, int maxX, int minY, int maxY, cv::Mat& returnImage);
void padImage(cv::Mat& image, int rows, int cols, cv::Mat& returnImage);
std::map<std::string, cv::Mat> readImagesFromDirectory(const std::string& directory);
long getMemoryUsage();

#endif
