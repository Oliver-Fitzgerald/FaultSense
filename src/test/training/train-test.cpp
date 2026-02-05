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

    std::map<std::string, std::array<float, 5>> anomaly;
    trainCell(anomaly);

    // Regression Test Cases
    CHECK( anomaly["chewinggum"][0] == Catch::Approx(15.8476));
    CHECK( anomaly["chewinggum"][1] == Catch::Approx(19.2713));
    CHECK( anomaly["chewinggum"][2] == Catch::Approx(13.0697));
    CHECK( anomaly["chewinggum"][3] == Catch::Approx(12.0211));
    CHECK( anomaly["chewinggum"][4] == Catch::Approx(39.7906));


    CHECK( anomaly["candle"][0] == Catch::Approx(14.0767));
    CHECK( anomaly["candle"][1] == Catch::Approx(19.2643));
    CHECK( anomaly["candle"][2] == Catch::Approx(14.1739));
    CHECK( anomaly["candle"][3] == Catch::Approx(14.2364));
    CHECK( anomaly["candle"][4] == Catch::Approx(38.2493));

    CHECK( anomaly["capsules"][0] == Catch::Approx(34.0935));
    CHECK( anomaly["capsules"][1] == Catch::Approx(9.09269));
    CHECK( anomaly["capsules"][2] == Catch::Approx(8.07414));
    CHECK( anomaly["capsules"][3] == Catch::Approx(11.4261));
    CHECK( anomaly["capsules"][4] == Catch::Approx(37.3158));

    CHECK( anomaly["cashew"][0] == Catch::Approx(26.3424));
    CHECK( anomaly["cashew"][1] == Catch::Approx(15.4123));
    CHECK( anomaly["cashew"][2] == Catch::Approx(9.50517));
    CHECK( anomaly["cashew"][3] == Catch::Approx(9.05762));
    CHECK( anomaly["cashew"][4] == Catch::Approx(39.7884));

    CHECK( anomaly["fryum"][0] == Catch::Approx(11.7872));
    CHECK( anomaly["fryum"][1] == Catch::Approx(21.5468));
    CHECK( anomaly["fryum"][2] == Catch::Approx(11.6668));
    CHECK( anomaly["fryum"][3] == Catch::Approx(12.3243));
    CHECK( anomaly["fryum"][4] == Catch::Approx(42.677));

    CHECK( anomaly["macaroni1"][0] == Catch::Approx(16.1601));
    CHECK( anomaly["macaroni1"][1] == Catch::Approx(20.2713));
    CHECK( anomaly["macaroni1"][2] == Catch::Approx(13.1044));
    CHECK( anomaly["macaroni1"][3] == Catch::Approx(13.9308));
    CHECK( anomaly["macaroni1"][4] == Catch::Approx(36.5345));

    CHECK( anomaly["macaroni2"][0] == Catch::Approx(10.5742));
    CHECK( anomaly["macaroni2"][1] == Catch::Approx(22.7598));
    CHECK( anomaly["macaroni2"][2] == Catch::Approx(10.8057));
    CHECK( anomaly["macaroni2"][3] == Catch::Approx(10.7872));
    CHECK( anomaly["macaroni2"][4] == Catch::Approx(45.0746));

    CHECK( anomaly["pcb1"][0] == Catch::Approx(24.6812));
    CHECK( anomaly["pcb1"][1] == Catch::Approx(13.9239));
    CHECK( anomaly["pcb1"][2] == Catch::Approx(13.3475));
    CHECK( anomaly["pcb1"][3] == Catch::Approx(14.167));
    CHECK( anomaly["pcb1"][4] == Catch::Approx(33.8823));

    CHECK( anomaly["pcb2"][0] == Catch::Approx(36.1703));
    CHECK( anomaly["pcb2"][1] == Catch::Approx(6.26119));
    CHECK( anomaly["pcb2"][2] == Catch::Approx(7.06122));
    CHECK( anomaly["pcb2"][3] == Catch::Approx(7.02788));
    CHECK( anomaly["pcb2"][4] == Catch::Approx(43.4741));

    CHECK( anomaly["pcb3"][0] == Catch::Approx(10.1761));
    CHECK( anomaly["pcb3"][1] == Catch::Approx(23.1765));
    CHECK( anomaly["pcb3"][2] == Catch::Approx(10.2964));
    CHECK( anomaly["pcb3"][3] == Catch::Approx(10.6853));
    CHECK( anomaly["pcb3"][4] == Catch::Approx(45.6671));

    CHECK( anomaly["pcb4"][0] == Catch::Approx(33.1674));
    CHECK( anomaly["pcb4"][1] == Catch::Approx(5.95831));
    CHECK( anomaly["pcb4"][2] == Catch::Approx(8.01386));
    CHECK( anomaly["pcb4"][3] == Catch::Approx(5.08332));
    CHECK( anomaly["pcb4"][4] == Catch::Approx(47.779));

    CHECK( anomaly["pipe_fryum"][0] == Catch::Approx(13.1748));
    CHECK( anomaly["pipe_fryum"][1] == Catch::Approx(22.435));
    CHECK( anomaly["pipe_fryum"][2] == Catch::Approx(6.65125));
    CHECK( anomaly["pipe_fryum"][3] == Catch::Approx(8.02478));
    CHECK( anomaly["pipe_fryum"][4] == Catch::Approx(49.7898));
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
