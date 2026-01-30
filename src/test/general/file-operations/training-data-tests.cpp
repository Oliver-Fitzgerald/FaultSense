/*
 * training-data-tests.cpp
 * Contains a collection of unit tests for functions contained in 
 * the corresponding file (training-data.cpp)
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
#include "../../../main/general/file-operations/training-data.h"

TEST_CASE ( "Reading and writing matrixs" ) {

    // Creating sample matrix
    cv::Mat sampleMatrix = cv::Mat(3, 3, CV_32SC1);
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            sampleMatrix.at<int>(row, col) = row + col;
        }
    }
    std::map<std::string, cv::Mat> original = { {"test", sampleMatrix} };

    // Writing sample matrix to memory as a normal sample
    writeNorm(original); 

    // Reading sample matrix from memory as a normal sample
    std::map<std::string, cv::Mat> read = { {"test", cv::Mat()} };
    readNorm(read);

    // Testing that the loaded matrix is identical to the original image
    cv::Mat readMatrix = read["test"];
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            REQUIRE( readMatrix.at<int>(row, col) == row + col );
        }
    }
}

TEST_CASE ( "Reading and writing distributions" ) {

    std::array<float, 5> original = {0.0, 0.1, 0.2, 0.3, 0.4};

    std::map<std::string, std::array<float, 5>> write = {{"test", original}};
    writeDistributions(write);

    std::map<std::string, std::array<float, 5>> read;
    readDistributions(read);

    for (int index = 0; index < 5; index++)
        REQUIRE_THAT (read["test"][index], Catch::Matchers::WithinAbs(original[index], 0.01) );

}
