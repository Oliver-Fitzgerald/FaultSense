#ifndef PreProcessing_H
#define PreProcessing_H

void illuminationInvariance(cv::Mat &image, cv::Mat &returnImage);
void checkFaultLBP(int (&normalSampe)[5], int (&anomolySample)[5], cv::Mat &image);

#endif
