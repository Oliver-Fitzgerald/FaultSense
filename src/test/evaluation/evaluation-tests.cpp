/*
 * evaluation-tests
 * Contains tests for functions in the corresponding main file
 * (evaluation.cpp)
 */

// Catch2
#include <catch2/catch_test_macros.hpp>
// Fault Sense
#include "../../main/evaluation/evaluation.h"
#include "../../main/feature/object-detection.h"
#include "../../main/training/train.h"
#include "../../main/objects/PreProcessing.h"

TEST_CASE ( "Evaluation interface test" ) {

    std::cout << "Generating normal samples ...\n";
    std::map<std::string, cv::Mat> normal;
    PreProcessing preProcessingConfiguration;
    preProcessingConfiguration.edge = true;
    trainMatrix(normal, preProcessingConfiguration, true);

    std::cout << "Generating anomaly samples ...\n";
    std::map<std::string, std::array<float, 5>> anomaly;
    trainCellNorms(anomaly, preProcessingConfiguration, false);

    std::cout << "Evaluating samples normal dataset instances ...\n";
    evaluateNormal("chewinggum", normal["chewinggum"], anomaly["chewinggum"]);
}
