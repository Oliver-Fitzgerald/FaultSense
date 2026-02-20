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

namespace {
    int countWhitePixels(cv::Mat &image);
}

TEST_CASE ("removeNoise test square") {

    // Image 000.JPG
    // WWWW
    // WBBW
    // WBBW
    // WWWW
    for (int minGrpSize = 0; minGrpSize < 10; minGrpSize++) {

        cv::Mat image = cv::imread("../../../data/test-images/synthetic/removeNoise/000.png");

        cv::Mat greyScale;
        cv::cvtColor(image, greyScale, cv::COLOR_BGR2GRAY);

        removeNoise(greyScale, minGrpSize);
        //imageViewer(greyScale);


        int whitePixelCount = countWhitePixels(greyScale);
        std::cout << "[removeNoise test square] TEST_CASE #" << minGrpSize  + 1<< ": minGrpSize(" << minGrpSize << "), whitePixelCount: (" << whitePixelCount << ")\n";
        if (minGrpSize < 5)
            CHECK(whitePixelCount == 4);
        else
            CHECK(whitePixelCount == 0);
    }
}

TEST_CASE ("removeNoise test V") {

    // Image 001.JPG
    // WWWWWWW
    // WBBWBBW
    // WWBBBWW
    // WWWWWWW
    for (int minGrpSize = 0; minGrpSize < 10; minGrpSize++) {

        cv::Mat image = cv::imread("../../../data/test-images/synthetic/removeNoise/001.png");

        cv::Mat greyScale;
        cv::cvtColor(image, greyScale, cv::COLOR_BGR2GRAY);

        // left off here
        removeNoise(greyScale, minGrpSize);
        //imageViewer(greyScale);


        int whitePixelCount = countWhitePixels(greyScale);
        std::cout << "[removeNoise test V] TEST_CASE #" << minGrpSize  + 1<< ": minGrpSize(" << minGrpSize << "), whitePixelCount: (" << whitePixelCount << ")\n";
        if (minGrpSize < 8)
            CHECK(whitePixelCount == 7);
        else
            CHECK(whitePixelCount == 0);
    }
}

TEST_CASE ("removeNoise test inverted V") {

    // Image 002.JPG
    // WWWWWWW
    // WWBBBWW
    // WBBWBBW
    // WWWWWWW
    for (int minGrpSize = 0; minGrpSize < 10; minGrpSize++) {

        cv::Mat image = cv::imread("../../../data/test-images/synthetic/removeNoise/002.png");

        cv::Mat greyScale;
        cv::cvtColor(image, greyScale, cv::COLOR_BGR2GRAY);

        // left off here
        removeNoise(greyScale, minGrpSize);
        //imageViewer(greyScale);


        int whitePixelCount = countWhitePixels(greyScale);
        std::cout << "[removeNoise test inverted V] TEST_CASE #" << minGrpSize  + 1<< ": minGrpSize(" << minGrpSize << "), whitePixelCount: (" << whitePixelCount << ")\n";
        if (minGrpSize < 8)
            CHECK(whitePixelCount == 7);
        else
            CHECK(whitePixelCount == 0);
    }
}

TEST_CASE ("removeNoise test image 4") {

    // Image 003.JPG
    // WWWWWWW
    // WBBBBBW
    // WBWWWBW
    // WBBBBBW
    // WWWWWWW
    for (int minGrpSize = 0; minGrpSize < 20; minGrpSize++) {

        cv::Mat image = cv::imread("../../../data/test-images/synthetic/removeNoise/003.png");

        cv::Mat greyScale;
        cv::cvtColor(image, greyScale, cv::COLOR_BGR2GRAY);

        // left off here
        removeNoise(greyScale, minGrpSize);
        imageViewer(greyScale);


        int whitePixelCount = countWhitePixels(greyScale);
        std::cout << "[removeNoise test image 4] TEST_CASE #" << minGrpSize  + 1<< ": minGrpSize(" << minGrpSize << "), whitePixelCount: (" << whitePixelCount << ")\n";
        if (minGrpSize < 13)
            CHECK(whitePixelCount == 12);
        else
            CHECK(whitePixelCount == 0);
    }
}

namespace {

    /*
     * whitePixelCount
     * Counts the number of white pixels in the passed image
     * @param image The image that the white pixels will be counted in
     */
    int countWhitePixels(cv::Mat &image) {

        int whitePixelCount = 0;
        for (int rows = 0; rows < image.rows; rows++) {
            for (int cols = 0; cols < image.cols; cols++) {
                
                int pixel = image.at<uchar>(rows, cols);
                if (pixel == 255)
                    whitePixelCount++;
            }
        }

        return whitePixelCount;
    }
}
