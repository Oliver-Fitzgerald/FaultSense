#ifndef train_H
#define train_H

#include <opencv2/opencv.hpp>
#include <map>
#include <string> 
#include <array>

void trainCellNorms(std::map<std::string, std::array<float, 5>> &cellNorm, const bool normal, const std::string& category = "");
void trainMatrix(std::map<std::string, cv::Mat> &matrixNorm);

#endif
