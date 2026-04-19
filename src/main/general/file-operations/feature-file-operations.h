#ifndef training_data_H
#define training_data_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <map>
#include <string>

void writeCellDistributions(std::map<std::string, std::array<float, 5>> &distributions);
void readCellDistributions(std::map<std::string, std::array<float, 5>> &distributions);
void writeMatrixNorm(std::map<std::string, cv::Mat> &norms); 
void readMatrixNorm(std::map<std::string, cv::Mat> &norms);

#endif
