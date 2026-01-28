#ifndef train_H
#define train_H

#include <opencv2/opencv.hpp>
#include <map>
#include <string> 
#include <array>

void trainAnomaly(std::map<std::string, std::array<float, 5>> &anomalyNorm);
void trainNormal(std::map<std::string, cv::Mat> &normalNorm);

#endif
