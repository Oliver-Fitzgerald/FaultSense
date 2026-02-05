/*
 * pre-processing-utils-tests
 * Contains unit tests for the coresponding functions in pre-processing-utils
 */

// Catch2 Testing Framework
#include <catch2/catch_test_macros.hpp>
// Fault Sense
#include "../../../main/feature/utils/pre-processing-utils.cpp"
// OpenCV2
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

TEST_CASE ("removeNoise test") {

    int maxGrpSize = 0;
    cv::Mat image = cv::imread("../../../data/synthetic/testing-data/remove-noise-1.JPG");

    removeNoise(image, maxGrpSize);
}
