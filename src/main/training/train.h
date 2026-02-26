#ifndef train_H
#define train_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <map>
#include <string> 
#include <array>
// Fault Sense
#include "../objects/PreProcessingPipeline.h"

void trainCellNorms(std::map<std::string, std::array<float, 5>> &cellNorms, PreProcessingPipeline &preProcessingConfiguration, const bool normal);
void trainMatrix(std::map<std::string, cv::Mat> &matrixNorm, PreProcessingPipeline &preProcessingConfiguration, const bool normal);

#endif
