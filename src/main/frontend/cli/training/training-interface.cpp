/*
 * training-interface
 * Contains the interface for the co-ordination/configuration of training processes
 */

// Standard
#include <map>
#include <string>
// Fault Sense
#include "../../../objects/PreProcessingPipeline.h"
#include "../../../training/train.h"
#include "../../../general/file-operations/training-data.h"

/*
 * train
 * Under Construction
 */
void train(std::map<std::string, bool> flags, PreProcessingPipeline& preProcessingPipeline) {

    std::cout << "Generate nomral norm matrix\n";
    std::map<std::string, cv::Mat> normalNorm = {{"chewinggum", cv::Mat()}};
    trainMatrix(normalNorm, preProcessingPipeline, true);
    std::cout << "Write normal norm to file\n";
    writeMatrixNorm(normalNorm); 

    std::cout << "Generate anomaly norm cell\n";
    std::map<std::string, std::array<float, 5>> anomalyNorm = {{"chewinggum", std::array<float,5>()}};
    trainCellNorms(anomalyNorm, preProcessingPipeline, false);
    std::cout << "Write anomaly norm to file\n";
    writeCellDistributions(anomalyNorm);

}
