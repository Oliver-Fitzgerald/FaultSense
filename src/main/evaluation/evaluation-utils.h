#ifndef evaluation_evaluate_utils_H
#define evaluation_evaluate_utils_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <map>
#include <array>
#include <string>
// Fault Sense
#include "../objects/EvaluationMetrics.h"
#include "../objects/FeaturesCollection.h"

namespace evaluate_utils {

    void initMatrix(const std::map<std::string, cv::Mat>::iterator &iterator, cv::Mat &categoryNorm);
    void initMatrix(const cv::Mat &sampleImage, cv::Mat &categoryNorm);
    bool evaluateImage(cv::Mat &image, FeaturesCollection& features, cv::Mat &normalMatrix, std::array<float, 5> &anomalySample, EvaluationMetrics& evaluationMetrics);

    int euclidianDistance(std::array<float, 5>& pointOne, std::array<float, 5>& pointTwo);
    int euclidianDistance(std::array<float, 5>& pointOne, float* pointTwo);

    int countWhitePixels(cv::Mat &image);

}

#endif
