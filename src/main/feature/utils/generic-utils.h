#ifndef METHODS_H
#define METHODS_H

#include "../objects/RGB.h"

void markFault(cv::Mat& image, int minX, int maxX, int minY, int maxY, const char* label = nullptr, RGB colour  = RGB{255,0,0});
void crop(cv::Mat& image, int minX, int maxX, int minY, int maxY, cv::Mat& returnImage);
void padImage(cv::Mat& image, int rows, int cols, cv::Mat& returnImage);

#endif
