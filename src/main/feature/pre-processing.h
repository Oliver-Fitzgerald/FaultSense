#ifndef pre_processing_functions_H
#define pre_processing_functions_H

void markFaultLBP(const std::array<float, 5>&normalSampe, const std::array<float, 5>& anomolySample, cv::Mat &image);
void illuminationInvariance(const cv::Mat &image, cv::Mat &returnImage);

#endif
