#ifndef PreProcessing_H
#define PreProcessing_H

void illuminationInvariance(cv::Mat &image, cv::Mat &returnImage);
void checkFaultLBP(float (&normalSampe)[5], float (&anomolySample)[5], cv::Mat &image);

#endif
