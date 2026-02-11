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

TEST_CASE ( "Evaluation interface test" ) {

    std::cout << "Generating normal samples ...\n";
    std::map<std::string, cv::Mat> normalNorm;
    trainMatrix(normalNorm);

    std::cout << "Generating anomaly samples ...\n";
    std::map<std::string, std::array<float, 5>> anomalyNorm;
    trainCellNorms(anomalyNorm, false, "chewinggum");

    std::cout << "Evaluating samples normal dataset instances ...\n";
    evaluateNormal("chewinggum", normalNorm["chewinggum"], anomalyNorm["chewinggum"]);
}
