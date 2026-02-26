#ifndef pre_processing_utils_internal
#define pre_processing_utils_internal

namespace internal {

    void clean(pixelGroup &grp, cv::Mat &img, int minGrpSize);
    cv::Mat brigthenDarkerAreas(const cv::Mat& img, const int threshold, const int amount);
    bool mergeOverlappingGroups(pixelGroup &currentGroup, std::vector<pixelGroup> &pixelGroups, std::vector<bool> &grpUsed, int row);
}

#endif
