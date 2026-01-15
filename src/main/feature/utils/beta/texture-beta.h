#ifndef TextureBeta_H
#define TextureBeta_H

void lbpValueDistribution(const cv::Mat &LVPValues, std::array<float, 5>& LBPHistogram);
void lbpValues(const cv::Mat &image, cv::Mat &LBPValues);

#endif
