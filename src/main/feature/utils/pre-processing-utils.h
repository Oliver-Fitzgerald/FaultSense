#include "../objects/HSV.h"
#include "../objects/CannyThreshold.h"
#include "../objects/PixelCoordinates.h"
void thresholdHSV(cv::Mat& image, HSV& threshold);
void edgeDetection(cv::Mat& image, cv::Mat& kernal, CannyThreshold& threshold);
void removeNoise(cv::Mat& img, int maxGrpSize);
