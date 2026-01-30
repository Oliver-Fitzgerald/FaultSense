#ifndef training_data_H
#define training_data_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <map>
#include <string>

void writeDistributions(std::map<std::string, std::array<float, 5>> &distributions);
void readDistributions(std::map<std::string, std::array<float, 5>> &distributions);
void writeNorm(std::map<std::string, cv::Mat> &norms); 
void readNorm(std::map<std::string, cv::Mat> &norms);

#endif
