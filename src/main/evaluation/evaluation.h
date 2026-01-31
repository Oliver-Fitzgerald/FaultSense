#ifndef evaluation_H
#define evaluation_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <array>
#include <string>

void evaluateNormal(const char *objectCategory, cv::Mat &normalNormSample, std::array<float, 5> &anomalyDistribution);

#endif
