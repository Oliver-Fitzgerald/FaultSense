/*
 * training-interface
 * Contains the interface for the co-ordination/configuration of training processes
 */

// Standard
#include <map>
#include <string>
// Fault Sense
#include "../../../objects/PreProcessingPipeline.h"
#include "../../../objects/FeaturesCollection.h"
#include "../../../training/train.h"
#include "../../../general/file-operations/feature-file-operations.h"

/*
 * train
 * Under Construction
 */
void train(std::map<std::string, bool> flags, FeaturesCollection& features) {

    int skipped = 0;
    for (const auto& [objectCategory, evaluate] : flags) {

        if (!evaluate) { skipped++; continue;}

        std::cout << "\033[32mINFO\033[0m: Generating nomral features\n";
        std::map<std::string, cv::Mat> normalFeatures;
        features.train(objectCategory, "Normal", true).extract(normalFeatures).reset();

        std::cout << "\033[32mINFO\033[0m: Writng normal features to file\n";
        writeObjectFeatures(normalFeatures, objectCategory, true);


        std::cout << "\033[32mINFO\033[0m: Generating anomaly feature\n";
        std::map<std::string, cv::Mat> anomalyFeatures;
        features.train(objectCategory, "Anomaly", true).extract(anomalyFeatures).reset();

        std::cout << "\033[32mINFO\033[0m: Writing anomaly feature to file ...\n";
        writeObjectFeatures(anomalyFeatures, objectCategory, false);

    }

    if (skipped == flags.size()) {
        std::cout << "WARNING: No object category was selected for training please choose one of the following object categories\n"
                  << "--chewinggum\n"
                  << "--cashew\n";
    }
}
