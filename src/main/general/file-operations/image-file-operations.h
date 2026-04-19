#ifndef generic_read_write_H
#define generic_read_write_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <string>
#include <filesystem>
#include <map>

void readImagesFromDirectory(const std::string& directory, std::map<std::string, cv::Mat> &images);
void readImagesFromDirectory(const std::string& directory, std::vector<cv::Mat> &images);
void writeImage(const cv::Mat &image, const std::string& dataDirectory);

#endif
