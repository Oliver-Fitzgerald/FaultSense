/*
 * Features-tests
 * Contains unit tests for Feature classes contained in the Features.cpp
 *
 * Feature Classes:
 * BinaryCountFeature - TODO
 * BinaryDistributionFeature - TODO
 */

// Catch2 Testing Framework
#include <catch2/catch_test_macros.hpp>
// OpenCV2
#include <opencv2/opencv.hpp>
// Fault Sense
#include "../../main/objects/Features.h"

/* validates that the correct number of pixels are counted in an image*/
TEST_CASE ("BinaryCountFeature", "[extractFeature]") {


    // Square image thresholded to 0 | 255
    cv::Mat image000 = cv::imread("test-images/synthetic/features/000.png");
    // Square image thresholded with values not thresholded
    cv::Mat image001 = cv::imread("test-images/synthetic/features/001.png");

}

TEST_CASE ("BinaryCountFeature", "[updateFeature]") {
}

TEST_CASE ("BinaryCountFeature", "[compareFeature]") {
}

TEST_CASE ("BinaryDistributionFeature", "[extractFeature]") {
}

TEST_CASE ("BinaryDistributionFeature", "[updateFeature]") {
}

TEST_CASE ("BinaryDistributionFeature", "[compareFeature]") {
}
