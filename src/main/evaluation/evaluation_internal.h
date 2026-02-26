#ifndef evaluation_internal_H
#define evaluation_internal_H

namespace internal {

    void initMatrix(const std::map<std::string, cv::Mat>::iterator &iterator, cv::Mat &categoryNorm);
    void initMatrix(const cv::Mat &sampleImage, cv::Mat &categoryNorm);
    bool evaluateImage(cv::Mat &image, cv::Mat &normalMatrix, std::array<float, 5> &anomalySample);

    int cellSize = 60;

    int averageNormalCells = 0;
    int averageAnomalyCells = 0;

}

#endif
