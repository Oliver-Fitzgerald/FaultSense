#ifndef evaluation_H
#define evaluation_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <array>
#include <string>
// Fault Sense
#include "../objects/PreProcessingPipeline.h"
#include "../objects/Features.h"

void evaluateObjectCategory(const char *objectCategory, cv::Mat &normalNormSample, std::array<float, 5> &anomalyDistribution, PreProcessingPipeline &preProcessingPipeline);
void markFaultLBP(const std::array<float, 5>&normalSampe, const std::array<float, 5>& anomolySample, cv::Mat &image);
void markFaultLBP(FeatureFilter& cellFeature, PreProcessingPipeline& preProcessingPipeline, cv::Mat& normalSample, const std::array<float, 5>& anomolySample, cv::Mat &image);
bool checkIfCellIsNormal(cv::Mat cell);

#endif
