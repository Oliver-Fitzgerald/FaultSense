#ifndef evaluation_H
#define evaluation_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <array>
#include <string>
// Fault Sense
#include "../objects/ConfusionMatrix.h"

void evaluateNormal(const char *objectCategory, cv::Mat &normalNormSample, std::array<float, 5> &anomalyDistribution);

bool evaluate(ConfusionMatrix& confusionMatrix, std::string& category, std::vector<std::array<float,5>>& normalSample, std::vector<std::array<float,5>>& anomalySample, cv::Mat& image, cv::Mat& imageMask);
bool evaluate(ConfusionMatrix& confusionMatrix, std::string& category, std::vector<std::array<float,5>>& normalSample, std::vector<std::array<float,5>>& anomalySample, cv::Mat& image);

#endif
