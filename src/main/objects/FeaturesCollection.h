#ifndef FeaturesPipeline_H
#define FeaturesPipeline_H

// Standard
#include <vector>
// Fault Sense
#include "Features.h"
#include "PreProcessingPipeline.h"

class FeaturesCollection {

public:
    std::map<std::unique_ptr<FeatureFilter>, std::unique_ptr<PreProcessingPipeline>> features;

    /*
     * getFeatureNames
     * populates a list of feature namese as strings
     * @param featureNames - the vector of feature names to be populated
     */
    void getFeatureNames(std::vector<std::string>& featureNames) {
        for (const auto& [filter, pipeline] : features)
            featureNames.push_back(filter->getName());
    }

    friend std::ostream& operator<<(std::ostream& os, const FeaturesCollection& features) {
        os << "Features Configuration\n\n";

        

        for (const auto& [feature, preProcessingPipeline] : features.features) {
            if (typeid(*feature) == typeid(BinaryCountFeature))
                os << "Collecting Feature BinaryCountFeature: true\n";
            else if (typeid(*feature) == typeid(BinaryDistributionFeature))
                os << "Collecting Feature BinaryDistributionFeature: true\n";
        }

        return os;
    }
};

#endif
