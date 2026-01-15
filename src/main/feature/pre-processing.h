#ifndef PreProcessing_H
#define PreProcessing_H

void markFaultLBP(const std::array<float, 5>&normalSampe, const std::array<float, 5>& anomolySample, cv::Mat &image);
void illuminationInvariance(const cv::Mat &image, cv::Mat &returnImage);

#endif
