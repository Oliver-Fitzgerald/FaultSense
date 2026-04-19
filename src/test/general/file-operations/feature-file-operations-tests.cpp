/*
 * feature-file-operations-tests.cpp
 * Contains a collection of unit tests for functions contained in 
 * the corresponding file (feature-file-operations.cpp)
 */

// Catch2
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <map>
#include <array>
#include <string>
// Fault Sense
#include "../../../main/general/file-operations/feature-file-operations.h"
#include "../../../global-variables.h"

TEST_CASE ( "Reading and writing matrixs" ) {

    global::projectRoot = "../../../";

    // Creating sample matrix
    cv::Mat sampleMatrix = cv::Mat(3, 3, CV_32SC1);
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            sampleMatrix.at<int>(row, col) = row + col;
        }
    }

    std::map<std::string, cv::Mat> original = { {"dummy-feature", sampleMatrix} };
    std::string objectCategory = "testing";

    // Writing sample matrix to memory as a normal sample
    writeObjectFeatures(original, objectCategory, true); 

    // Reading sample matrix from memory as a normal sample
    std::map<std::string, cv::Mat> read = { {"dummy-feature", cv::Mat()} };
    readObjectFeatures(read, objectCategory, true);

    // Testing that the loaded matrix is identical to the original image
    cv::Mat readMatrix = read["dummy-feature"];
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            REQUIRE( readMatrix.at<int>(row, col) == row + col );
        }
    }
}
