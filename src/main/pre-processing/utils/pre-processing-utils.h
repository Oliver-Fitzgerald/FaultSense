#ifndef pre_processing_utils_H
#define pre_processing_utils_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Fault Sense
#include "../../objects/PixelCoordinates.h"

cv::Mat brigthenDarkerAreas(const cv::Mat& img, const int threshold, const int amount);
void binaryThreshold(cv::Mat& image, int threshold);
void illuminationInvariance(const cv::Mat &image, cv::Mat &returnImage);

namespace pre_processing_utils {

    bool mergeOverlappingGroups(PixelGroup &currentGroup, std::vector<PixelGroup> &pixelGroups, std::vector<bool> &grpUsed, int row);
    void clean(PixelGroup &grp, cv::Mat &img, int minGrpSize);
    void initMatrix(const cv::Mat &sampleImage, cv::Mat &categoryNorm);
    uint8_t pixelLBP(const cv::Mat &image, const int x, const int y);
}

#endif
