/*
 * train-test
 * Unit tests for train.cpp
 */

// Catch2 Testing Framework
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
// Fault Sense
#include "../../main/training/train.h"
// OpenCV2
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Standard
#include <map>
#include <string>
#include <array>


/*
 * void trainAnomaly(std::map<std::string, std::array<float, 5>> &anomaly);
 *
 * Regression Testing
 * As the LBP value distribution would be exesivley time consuming to calculate
 * by hand we perform regression testing instead as we should see the same result 
 * for the same data each time we run it. We will run tests on the lower level functions
 * that calculate LBP values to ensure that they themshelves are working correctly.
 * (TODO: Insert link here to documentation or something mayber??)
 */


TEST_CASE( "Training cell norm - regression", "[trainCell]" ) {

    const std::string objectCategories[12] = {
        "chewinggum",
        "candle",
        "capsules",
        "cashew",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum"
    };

    std::map<std::string, std::array<float, 5>> anomaly;
    trainCell(anomaly, false);

    for (int index = 0; index < 12; index++) {
            
        std::array<float, 5> categoryNorm = anomaly[ objectCategories[index] ];

        float total = 0;
        for (int index = 0; index < 5; index++)
            total += categoryNorm[index];

        REQUIRE_THAT( total, Catch::Matchers::WithinAbs(100,0.1) );
    }
}

TEST_CASE( "Training cell norm individual", "[trainCell]" ) {

    const std::string objectCategories[12] = {
        "chewinggum",
        "candle",
        "capsules",
        "cashew",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum"
    };

    for (int index = 0; index < 12; index++) {

        std::map<std::string, std::array<float, 5>> anomaly;
        trainCell(anomaly, false, objectCategories[index].c_str());

                
        std::array<float, 5> categoryNorm = anomaly[ objectCategories[index] ];

        float total = 0;
        for (int index = 0; index < 5; index++)
            total += categoryNorm[index];

        REQUIRE_THAT( total, Catch::Matchers::WithinAbs(100,0.1) );
    }
}

TEST_CASE( "Training matrix norm distribution is normalized", "[trainMatrix]" ) {

    const std::string objectCategories[12] = {
        "chewinggum",
        "candle",
        "capsules",
        "cashew",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum"
    };

    std::map<std::string, cv::Mat> normal;
    trainMatrix(normal);


    for (int index = 0; index < 12; index++) {
            
        cv::Mat categoryNorm = normal[ objectCategories[index] ];


        for (int row = 0; row < categoryNorm.rows; row++) {
            for (int col = 0; col < categoryNorm.cols; col++) {

                float* lbpDistribution = categoryNorm.ptr<float>(row, col);

                float total = 0;
                for (int index = 0; index < 5; index++)
                    total += lbpDistribution[index];

                REQUIRE_THAT( total, Catch::Matchers::WithinAbs(100,0.1) );
            }
        }
    }


}
