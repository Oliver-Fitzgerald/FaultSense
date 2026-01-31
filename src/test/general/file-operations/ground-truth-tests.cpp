/*
 * ground-truth-tests
 * Contains the unit tests for the functinos in it's coresponding main file ground-truth.cpp
 */

// Catch2
#include <catch2/catch_test_macros.hpp>
// Fault Sense
#include "../../../main/general/file-operations/ground-truth.h"
// Standard
#include <iomanip>

TEST_CASE ( "Correct chewinggum binary classification" ) {


    std::map<std::string, std::array<std::string, 5>> parsedObjectLabels;
    readVisaLabels("chewinggum", parsedObjectLabels);

    // Check Normal
    for (int index = 0; index < 502; index++) {

        std::string key;
        std::string string = std::to_string(index);
        if (string.length() < 3)
            key = std::string(3 - string.length(), '0') + string;
        else
            key = string;
        REQUIRE (parsedObjectLabels["N" + key + ".JPG"][0] == "normal" );
    }

    // Check Anomaly
    for (int index = 0; index < 99; index++) {

        std::string key;
        std::string string = std::to_string(index);
        if (string.length() < 3)
            key = std::string(3 - string.length(), '0') + string;
        else
            key = string;

        REQUIRE (parsedObjectLabels["A" + key + ".png"][0] != "normal" );
        REQUIRE (parsedObjectLabels["A" + key + ".png"][0] != "" );
    }
}
