#ifndef synthetic_data_generation_H
#define synthetic_data_generation_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <array>

void generateRemoveNoiseTestData(std::array<cv::Mat, 4> &testImages);
void generateRemoveNoiseTestData(std::array<cv::Mat, 5> &testImages);

#endif
