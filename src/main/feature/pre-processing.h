#ifndef PreProcessing_H
#define PreProcessing_H

void markFaultLBP(std::vector<std::array<float, 5>>&normalSampe, std::vector<std::array<float, 5>>& anomolySample, cv::Mat &image, std::string& imageCategory, const cv::Mat& imageMask);
void illuminationInvariance(const cv::Mat &image, cv::Mat &returnImage);

#endif
