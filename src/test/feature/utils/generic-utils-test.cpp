/*
 * generic-utils-test
 * Unit test cases for generic-utils
 */ 

// Catch2 Testing Framework
#include <catch2/catch_test_macros.hpp>
// Fault Sense
#include "../../../main/feature/utils/generic-utils.cpp"
// OpenCV2
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Standard Libary
#include <vector>
#include <iostream>
#include <stdexcept>

/*
 * void crop(cv::Mat& image, int minX, int maxX, int minY, int maxY, cv::Mat& returnImage);
 * Asserts cropping to within image bounds reutrns expected dimensions
 */
TEST_CASE( "Croping Image", "[crop]" ) {

    cv::Mat input = cv::imread("../../../data/test-images/chewinggum-anomoly.JPG");
    cv::Mat output;
    crop(input, 0, 10, 0, 10, output);
    REQUIRE( output.cols == 10 );
    REQUIRE( output.rows == 10 );
}

/*
 * void crop(cv::Mat& image, int minX, int maxX, int minY, int maxY, cv::Mat& returnImage);
 * Asserts attempting to increace image dimensions is invalid
 */
TEST_CASE( "Increace Image Size", "[crop]" ) {

    cv::Mat input = cv::imread("../../../data/test-images/chewinggum-anomoly.JPG");
    cv::Mat output;
    REQUIRE_THROWS_AS( crop(input, -10, 10, -10, 10, output), std::out_of_range );
    REQUIRE_THROWS_AS( crop(input, 20, 10, 20, 10, output), std::out_of_range );

}
