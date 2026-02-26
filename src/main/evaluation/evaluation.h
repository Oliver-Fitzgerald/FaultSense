#ifndef evaluation_H
#define evaluation_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <array>
#include <string>
// Fault Sense
#include "../objects/PreProcessingPipeline.h"

void evaluateObjectCategory(const char *objectCategory, cv::Mat &normalNormSample, std::array<float, 5> &anomalyDistribution, PreProcessingPipeline &preProcessingPipeline);

#endif
