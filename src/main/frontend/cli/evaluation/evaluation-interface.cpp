/*
 * evaluation-interface
 * Contains the interface for the co-ordination/configuration of evaluation processes
 */

// Standard
#include <map>
#include <string>
// Fault Sense
#include "../../../objects/PreProcessingPipeline.h"
#include "../../../objects/FeaturesCollection.h"
#include "../../../general/file-operations/feature-file-operations.h"
#include "../../../evaluation/evaluation.h"

/*
 * evaluation
 * Under Construction
 */
void evaluation(std::map<std::string, bool>& flags, FeaturesCollection& features) {

    std::vector<std::string> featureNames;
    features.getFeatureNames(featureNames);

    int skipped = 0;
    for (const auto& [objectCategory, evaluate] : flags) {

        if (!evaluate) { skipped++; continue;}

        try {

            std::cout << "Reading trained " << objectCategory << " normal features ...\n";
            std::map<std::string, cv::Mat> normalFeatures;
            for (const auto& feature : featureNames)
                normalFeatures[feature] = cv::Mat();
            readObjectFeatures(normalFeatures, objectCategory, true);

            std::cout << "Reading trained " << objectCategory << " anomaly features ...\n";
            std::map<std::string, cv::Mat> anomalyFeatures;
            for (const auto& feature : featureNames)
                anomalyFeatures[feature] = cv::Mat();
            readObjectFeatures(anomalyFeatures, objectCategory, false);

            std::cout << "Evaluating " << objectCategory << " instances ...\n";
            evaluateObjectCategory(objectCategory, features, normalFeatures, anomalyFeatures);

        } catch (const std::exception& e) {

            std::cerr << "\033[31mFATAL ERROR\033[0m: " << e.what() << '\n';
            std::cerr << "\033[93mSuggestion\033[0m: Have you trained features for this object category?\n";
            std::exit(EXIT_FAILURE);
        }
    }

    if (skipped == flags.size()) {
        std::cout << "WARNING: No object category was selected to be evaluated please choose one of the following flags\n"
                  << "--chewinggum\n"
                  << "--cashew\n";
    }
}
