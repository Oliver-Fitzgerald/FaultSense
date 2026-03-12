#ifndef feature_extration_H
#define feature_extration_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <array>

void lbpValueDistribution(const cv::Mat &LVPValues, std::array<float, 5>& LBPHistogram);

#endif
