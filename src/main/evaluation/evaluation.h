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
#include "../objects/EvaluationMetrics.h"

void evaluateObjectCategory(const std::string& objectCategory, FeaturesCollection& features, std::map<std::string, cv::Mat>& normalFeatures, std::map<std::string, cv::Mat>& anomalyFeatures);

void markFaultLBP(const std::array<float, 5>& normalSample, const std::array<float, 5>& anomolySample, cv::Mat &image);
void markFaults(FeaturesCollection& normalFeatures, FeaturesCollection& anomalyFeatures, cv::Mat &image);

bool checkIfCellIsNormal(cv::Mat cell);

#endif
