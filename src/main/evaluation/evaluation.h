#ifndef evaluation_H
#define evaluation_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <array>
#include <string>
// Fault Sense
#include "../objects/PreProcessingPipeline.h"
#include "../objects/FeaturesCollection.h"
#include "../objects/Features.h"

void evaluateObjectCategory(const char *objectCategory, cv::Mat &normalNormSample, std::array<float, 5> &anomalyDistribution, FeaturesCollection& features);
void markFaultLBP(const std::array<float, 5>& normalSample, const std::array<float, 5>& anomolySample, cv::Mat &image);
void markFaults(FeaturesCollection& normalFeatures, FeaturesCollection& anomalyFeatures, cv::Mat &image);

bool checkIfCellIsNormal(cv::Mat cell);

#endif
