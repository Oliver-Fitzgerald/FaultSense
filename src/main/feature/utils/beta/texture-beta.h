#ifndef TextureBeta_H
#define TextureBeta_H

void computeLBP(cv::Mat &image, cv::Mat &LBPValues, int (&LBPHistogram)[5]);
cv::Mat brigthen_darker_areas(const cv::Mat& img, int threshold, int amount);

#endif
