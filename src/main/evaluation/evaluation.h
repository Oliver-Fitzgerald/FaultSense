#ifndef evaluation_H
#define evaluation_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <array>
#include <string>
// Fault Sense
#include "../objects/EvaluationMetrics.h"
#include "../objects/PreProcessingPipeline.h"

class FeaturesCollection;

void evaluateObjectCategory(const std::string& objectCategory, FeaturesCollection& features, std::map<std::string, cv::Mat>& normalFeatures, std::map<std::string, cv::Mat>& anomalyFeatures);

void markFaultLBP(const std::array<float, 5>& normalSample, const std::array<float, 5>& anomolySample, cv::Mat &image);
void markFaults(std::map<std::string, cv::Mat>& normalFeatures, std::map<std::string, cv::Mat>& anomalyFeatures, cv::Mat &image, FeaturesCollection& featureCollection, std::string imageName);

bool checkIfCellIsNormal(cv::Mat cell);

#endif
