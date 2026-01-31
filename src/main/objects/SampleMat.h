/*
 * SampleMat
 * Contains the SampleMat class
 * The SampleMat class extends OpenCV2s Mat class to incorporate metadata about
 * a sample from the dataset into it's matrix implmentation for ease of access.
 */

#ifndef sample_mat_H
#define sample_mat_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <string>
#include <array>

class SampleMat : public cv::Mat {

public:
    using cv::Mat::Mat;

    std::string filename;
    bool normal;

    SampleMat(cv::Mat &matrix, std::string &filename, bool &normal, std::array<std::string, 5> &anomalyTypes)
        : cv::Mat(matrix), filename(filename), normal(normal), anomalyTypes(anomalyTypes) {}

private:
    std::array<std::string, 5> anomalyTypes;

};


#endif
