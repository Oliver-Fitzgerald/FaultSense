/*
 * pre-processing-utils-tests
 * Contains unit tests for the coresponding functions in pre-processing-utils
 */

// Catch2 Testing Framework
#include <catch2/catch_test_macros.hpp>
// Fault Sense
#include "../../../main/general/generic-utils.h"
#include "../../../main/feature/utils/pre-processing-utils.h"
#include "../../../main/feature/utils/pre-processing-utils_internal.h"
#include "../../../main/frontend/cli/image-viewer-ui/image-viewer.h"
// OpenCV2
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {
    int countWhitePixels(cv::Mat &image);
}

TEST_CASE ("removeNoise test square") {

    // Image 000.png
    // WWWW
    // WBBW
    // WBBW
    // WWWW
    std::cout << "Running test case [removeNoise test square]\n";
    cv::Mat image = cv::imread("../../../data/test-images/synthetic/removeNoise/000.png");

    cv::Mat greyScale;
    cv::cvtColor(image, greyScale, cv::COLOR_BGR2GRAY);

    for (int minGrpSize = 0; minGrpSize < 10; minGrpSize++) {

        std::cout << "Memory Usage Before: " << getMemoryUsage() << "\n";;
        removeNoise(greyScale, minGrpSize);

        int whitePixelCount = countWhitePixels(greyScale);
        std::cout << "[removeNoise test square] TEST_CASE #" << minGrpSize  + 1<< ": minGrpSize(" << minGrpSize << "), whitePixelCount: (" << whitePixelCount << ")\n";
        if (minGrpSize < 5)
            CHECK(whitePixelCount == 4);
        else
            CHECK(whitePixelCount == 0);
    }
}

TEST_CASE ("removeNoise test V") {

    // Image 001.png
    // WWWWWWW
    // WBBWBBW
    // WWBBBWW
    // WWWWWWW
    std::cout << "Running test case [removeNoise test V]\n";
    cv::Mat image = cv::imread("../../../data/test-images/synthetic/removeNoise/001.png");

    cv::Mat greyScale;
    cv::cvtColor(image, greyScale, cv::COLOR_BGR2GRAY);
    std::cout << "image greyscale: " << getMemoryUsage() << "\n";

    for (int minGrpSize = 0; minGrpSize < 10; minGrpSize++) {

        removeNoise(greyScale, minGrpSize);
        std::cout << "image removeNoise: " << getMemoryUsage() << "\n";

        int whitePixelCount = countWhitePixels(greyScale);
        std::cout << "[removeNoise test V] TEST_CASE #" << minGrpSize  + 1<< " COMPLETE: minGrpSize(" << minGrpSize << "), whitePixelCount: (" << whitePixelCount << ")\n";
        if (minGrpSize < 8)
            CHECK(whitePixelCount == 7);
        else
            CHECK(whitePixelCount == 0);
    }
}

TEST_CASE ("removeNoise test inverted V") {

    // Image 002.png
    // WWWWWWW
    // WWBBBWW
    // WBBWBBW
    // WWWWWWW
    std::cout << "Running test case [removeNoise test inverted V]\n";
    cv::Mat image = cv::imread("../../../data/test-images/synthetic/removeNoise/002.png");

    cv::Mat greyScale;
    cv::cvtColor(image, greyScale, cv::COLOR_BGR2GRAY);

    for (int minGrpSize = 0; minGrpSize < 10; minGrpSize++) {

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

TEST_CASE ("removeNoise test image 4") { // Image 003.png
    // WWWWWWW
    // WBBBBBW
    // WBWWWBW
    // WBBBBBW
    // WWWWWWW
    std::cout << "Running test case [removeNoise test image 4]\n";
    cv::Mat image = cv::imread("../../../data/test-images/synthetic/removeNoise/003.png");

    cv::Mat greyScale;
    cv::cvtColor(image, greyScale, cv::COLOR_BGR2GRAY);
    for (int minGrpSize = 0; minGrpSize < 20; minGrpSize++) {

        removeNoise(greyScale, minGrpSize);

        int whitePixelCount = countWhitePixels(greyScale);
        std::cout << "[removeNoise test image 4] TEST_CASE #" << minGrpSize  + 1<< ": minGrpSize(" << minGrpSize << "), whitePixelCount: (" << whitePixelCount << ")\n";
        if (minGrpSize < 13)
            CHECK(whitePixelCount == 12);
        else
            CHECK(whitePixelCount == 0);
    }
}


struct MergeOverlapData {

    std::vector<pixelGroup> pixelGroups = {
    {
        .group = {{1,1},{1,2}},
        .bounds = {{1, 2}},
        .row = 1
    }};
    std::vector<bool> grpUsed = {true};
};

TEST_CASE ("mergeOverlap") {

    MergeOverlapData data;

    SECTION("Equal Overlap") {

        /* Example
         * Image 000.png
         * WWWW
         * WBBW
         * WBBW -> testGroup
         * WWWW
         */
        pixelGroup testGroup = {
            .group = {{2,1},{2,2}},
            .bounds = {{1, 2}},
            .row = 2
        };
        int currentRow = 2;

        bool existingGroup = internal::mergeOverlappingGroups(testGroup, data.pixelGroups, data.grpUsed, currentRow);
        REQUIRE(existingGroup == true);
    }

    SECTION("Left Skewed") {

        /* Example
         * WWWW
         * WBBW
         * BBWW -> testGroup
         * WWWW
         */
        pixelGroup testGroup = {
            .group = {{2,0},{2,1}},
            .bounds = {{0, 1}},
            .row = 2
        };
        int currentRow = 2;

        bool existingGroup = internal::mergeOverlappingGroups(testGroup, data.pixelGroups, data.grpUsed, currentRow);
        REQUIRE(existingGroup == true);
    }
    SECTION("Rigth Skewed") {

        /* Example
         * WWWW
         * WBBW
         * WWBB -> testGroup
         * WWWW
         */
        pixelGroup testGroup = {
            .group = {{2,2},{2,3}},
            .bounds = {{2, 3}},
            .row = 2
        };
        int currentRow = 2;

        bool existingGroup = internal::mergeOverlappingGroups(testGroup, data.pixelGroups, data.grpUsed, currentRow);
        REQUIRE(existingGroup == true);
    }
    SECTION("Overflow") {

        /* Example
         * WWWW
         * WBBW
         * BBBB -> testGroup
         * WWWW
         */
        pixelGroup testGroup = {
            .group = {{2,0},{2,1},{2,2},{2,3}},
            .bounds = {{0, 3}},
            .row = 2
        };
        int currentRow = 2;

        bool existingGroup = internal::mergeOverlappingGroups(testGroup, data.pixelGroups, data.grpUsed, currentRow);
        REQUIRE(existingGroup == true);
    }
    SECTION("underflow") {

        /* Example
         * WWWW
         * BBBB
         * WBBW -> testGroup
         * WWWW
         */
        data.pixelGroups[0].group = {{1,0}, {1,1}, {1,2}, {1,3}};
        data.pixelGroups[0].bounds = {{0,3}};
        pixelGroup testGroup = {
            .group = {{2,1},{2,2}},
            .bounds = {{1, 2}},
            .row = 2
        };
        int currentRow = 2;

        bool existingGroup = internal::mergeOverlappingGroups(testGroup, data.pixelGroups, data.grpUsed, currentRow);
        REQUIRE(existingGroup == true);
    }
    SECTION("Disconnected") {

        /* Example
         * WWWW
         * BBBB
         * BWWB
         * BBBB
         */
        data.pixelGroups[0].group = {{1,0}, {1,1}, {1,2}, {1,3}};
        data.pixelGroups[0].bounds = {{0,3}};
        pixelGroup testGroupOne = {
            .group = {{2,0}},
            .bounds = {{0, 0}},
            .row = 2
        };
        pixelGroup testGroupTwo = {
            .group = {{2,3}},
            .bounds = {{3, 3}},
            .row = 2
        };
        int currentRow = 2;

        bool existingGroup = internal::mergeOverlappingGroups(testGroupOne, data.pixelGroups, data.grpUsed, currentRow);
        CHECK(existingGroup == true);

        data.pixelGroups[0].append(testGroupOne, currentRow);

        existingGroup = internal::mergeOverlappingGroups(testGroupTwo, data.pixelGroups, data.grpUsed, currentRow);
        CHECK(existingGroup == true);

        REQUIRE(existingGroup == true);

        currentRow  = 3;
        pixelGroup testGroupThree = {
            .group = {{3,0},{3,1},{3,2},{3,3}},
            .bounds = {{0, 3}},
            .row = 3
        };

        existingGroup = internal::mergeOverlappingGroups(testGroupThree, data.pixelGroups, data.grpUsed, currentRow);
        REQUIRE(existingGroup == true);
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
