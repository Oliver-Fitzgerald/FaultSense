#ifndef pre_processing_utils_internal_H
#define pre_processing_utils_internal_H

namespace pre_processing_utils {

    bool mergeOverlappingGroups(PixelGroup &currentGroup, std::vector<PixelGroup> &pixelGroups, std::vector<bool> &grpUsed, int row);
    void clean(PixelGroup &grp, cv::Mat &img, int minGrpSize);
    void initMatrix(const cv::Mat &sampleImage, cv::Mat &categoryNorm);
    uint8_t pixelLBP(const cv::Mat &image, const int x, const int y);

}
#endif
