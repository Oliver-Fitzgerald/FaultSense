#ifndef feature_extration_H
#define feature_extration_H

void lbpValueDistribution(const cv::Mat &LVPValues, std::array<float, 5>& LBPHistogram);
void lbpValues(const cv::Mat &image, cv::Mat &LBPValues);
bool checkIfCellIsNormal(cv::Mat cell);

#endif
