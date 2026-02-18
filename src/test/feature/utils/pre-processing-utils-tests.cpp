/*
 * pre-processing-utils-tests
 * Contains unit tests for the coresponding functions in pre-processing-utils
 */

// Catch2 Testing Framework
#include <catch2/catch_test_macros.hpp>
// Fault Sense
#include "../../../main/feature/utils/pre-processing-utils.h"
#include "../../../main/frontend/cli/image-viewer-ui/image-viewer.h"
// OpenCV2
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

TEST_CASE ("removeNoise test") {

    // Image 000.JPG
    // WWWW
    // WBBW
    // WBBW
    // WWWW
    int maxGrpSize = 40;
    cv::Mat image = cv::imread("../../../data/test-images/synthetic/removeNoise/000.JPG");

    removeNoise(image, maxGrpSize);
    imageViewer(image);
}
