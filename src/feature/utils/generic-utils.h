#ifndef METHODS_H
#define METHODS_H

void markFault(cv::Mat& image, int minX, int maxX, int minY, int maxY, const char* label);
void crop(cv::Mat& image, int minX, int maxX, int minY, int maxY, cv::Mat& returnImage);

#endif
