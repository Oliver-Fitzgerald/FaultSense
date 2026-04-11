/*
 * evaluation-interface
 * Contains the interface for the co-ordination/configuration of evaluation processes
 */

// Standard
#include <map>
#include <string>
// Fault Sense
#include "../../../objects/PreProcessingPipeline.h"
#include "../../../general/file-operations/training-data.h"
#include "../../../evaluation/evaluation.h"

/*
 * evaluation
 * Under Construction
 */
void evaluation(std::map<std::string, bool> flags, PreProcessingPipeline& preProcessingPipeline) {

    std::cout << "Reading normal norm ...\n";
    std::map<std::string, cv::Mat> normalNorm = {
        {"chewinggum", cv::Mat()},
        {"candle", cv::Mat()},
        {"capsules", cv::Mat()},
        {"cashew", cv::Mat()},
        {"fryum", cv::Mat()},
        {"macaroni1", cv::Mat()},
        {"macaroni2", cv::Mat()},
        {"pcb1", cv::Mat()},
        {"pcb2", cv::Mat()},
        {"pcb3", cv::Mat()},
        {"pcb4", cv::Mat()},
        {"pipe_fryum", cv::Mat()}
    };
    readMatrixNorm(normalNorm);

    std::cout << "Reading anomaly norm ...\n";
    std::map<std::string, std::array<float, 5>> anomalyNorm = {{"chewinggum", std::array<float, 5>()}};
    readCellDistributions(anomalyNorm);


    if (flags["chewinggum"]) {
        std::cout << "Evaluating chewing gum instances ...\n";
        evaluateObjectCategory("chewinggum", normalNorm["chewinggum"], anomalyNorm["chewinggum"], preProcessingPipeline);
    } else {
        std::cout << "Warning: No object type selected\n";
    }
}
