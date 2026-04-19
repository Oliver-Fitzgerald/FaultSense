#ifndef training_data_H
#define training_data_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <map>
#include <string>

void writeObjectFeatures(std::map<std::string, cv::Mat> &features, const std::string objectCategory, bool normal);
void readObjectFeatures(std::map<std::string, cv::Mat> &features, const std::string objectCategory, bool normal);

#endif
