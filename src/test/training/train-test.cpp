/*
 * train-test
 * Unit tests for train.cpp
 */

// Catch2 Testing Framework
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
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
TEST_CASE( "Training anomaly sample", "[trainAnomaly]" ) {

    std::map<std::string, std::array<float, 5>> anomaly;
    trainAnomaly(anomaly);

    // Regression Test Cases
    CHECK( anomaly["chewinggum"][0] == Catch::Approx(18.2672));
    CHECK( anomaly["chewinggum"][1] == Catch::Approx(18.9773));
    CHECK( anomaly["chewinggum"][2] == Catch::Approx(8.42583));
    CHECK( anomaly["chewinggum"][3] == Catch::Approx(10.0648));
    CHECK( anomaly["chewinggum"][4] == Catch::Approx(44.3411));


    CHECK( anomaly["candle"][0] == Catch::Approx(11.3254));
    CHECK( anomaly["candle"][1] == Catch::Approx(22.0251));
    CHECK( anomaly["candle"][2] == Catch::Approx(11.4596));
    CHECK( anomaly["candle"][3] == Catch::Approx(11.5096));
    CHECK( anomaly["candle"][4] == Catch::Approx(43.7352));


    CHECK( anomaly["capsules"][0] == Catch::Approx(31.3955));
    CHECK( anomaly["capsules"][1] == Catch::Approx(12.8982));
    CHECK( anomaly["capsules"][2] == Catch::Approx(11.5315));
    CHECK( anomaly["capsules"][3] == Catch::Approx(9.50337));
    CHECK( anomaly["capsules"][4] == Catch::Approx(34.4393));

    CHECK( anomaly["cashew"][0] == Catch::Approx(25.1507));
    CHECK( anomaly["cashew"][1] == Catch::Approx(15.5907));
    CHECK( anomaly["cashew"][2] == Catch::Approx(9.69751));
    CHECK( anomaly["cashew"][3] == Catch::Approx(9.63382));
    CHECK( anomaly["cashew"][4] == Catch::Approx(39.7872));

    CHECK( anomaly["fryum"][0] == Catch::Approx(12.1301));
    CHECK( anomaly["fryum"][1] == Catch::Approx(21.1009));
    CHECK( anomaly["fryum"][2] == Catch::Approx(12.0124));
    CHECK( anomaly["fryum"][3] == Catch::Approx(12.5329));
    CHECK( anomaly["fryum"][4] == Catch::Approx(42.2072));

    CHECK( anomaly["macaroni1"][0] == Catch::Approx(13.1679));
    CHECK( anomaly["macaroni1"][1] == Catch::Approx(21.5428));
    CHECK( anomaly["macaroni1"][2] == Catch::Approx(11.7432));
    CHECK( anomaly["macaroni1"][3] == Catch::Approx(12.4018));
    CHECK( anomaly["macaroni1"][4] == Catch::Approx(41.097));

    CHECK( anomaly["macaroni2"][0] == Catch::Approx(10.2912));
    CHECK( anomaly["macaroni2"][1] == Catch::Approx(22.932));
    CHECK( anomaly["macaroni2"][2] == Catch::Approx(10.3448));
    CHECK( anomaly["macaroni2"][3] == Catch::Approx(10.4284));
    CHECK( anomaly["macaroni2"][4] == Catch::Approx(46.0952));

    CHECK( anomaly["pcb1"][0] == Catch::Approx(36.4094));
    CHECK( anomaly["pcb1"][1] == Catch::Approx(6.8424));
    CHECK( anomaly["pcb1"][2] == Catch::Approx(7.72647));
    CHECK( anomaly["pcb1"][3] == Catch::Approx(7.60491));
    CHECK( anomaly["pcb1"][4] == Catch::Approx(41.6358));

    CHECK( anomaly["pcb2"][0] == Catch::Approx(35.2289));
    CHECK( anomaly["pcb2"][1] == Catch::Approx(7.37718));
    CHECK( anomaly["pcb2"][2] == Catch::Approx(8.94372));
    CHECK( anomaly["pcb2"][3] == Catch::Approx(7.6638));
    CHECK( anomaly["pcb2"][4] == Catch::Approx(40.9599));

    CHECK( anomaly["pcb3"][0] == Catch::Approx(13.4426));
    CHECK( anomaly["pcb3"][1] == Catch::Approx(21.0843));
    CHECK( anomaly["pcb3"][2] == Catch::Approx(11.2075));
    CHECK( anomaly["pcb3"][3] == Catch::Approx(11.4511));
    CHECK( anomaly["pcb3"][4] == Catch::Approx(42.8134));

    CHECK( anomaly["pcb4"][0] == Catch::Approx(32.3164));
    CHECK( anomaly["pcb4"][1] == Catch::Approx(6.74347));
    CHECK( anomaly["pcb4"][2] == Catch::Approx(9.21298));
    CHECK( anomaly["pcb4"][3] == Catch::Approx(6.66326));
    CHECK( anomaly["pcb4"][4] == Catch::Approx(45.1139));

    CHECK( anomaly["pipe_fryum"][0] == Catch::Approx(12.7631));
    CHECK( anomaly["pipe_fryum"][1] == Catch::Approx(22.3707));
    CHECK( anomaly["pipe_fryum"][2] == Catch::Approx(7.16383));
    CHECK( anomaly["pipe_fryum"][3] == Catch::Approx(8.36278));
    CHECK( anomaly["pipe_fryum"][4] == Catch::Approx(49.5458));
}
