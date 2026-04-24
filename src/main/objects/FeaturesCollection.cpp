/* FeaturesCollection.cpp
 */

#include "FeaturesCollection.h"

// Full definitions now included
#include "Features.h"
#include "PreProcessingPipeline.h"
#include "MODE.h"


// Fault Sense
#include "../../global-variables.h"
#include "../general/file-operations/image-file-operations.h"

// Standard
#include <stdexcept>
#include <typeinfo>
#include <vector>


/*
 * train
 * trains features for classification based on pre-processing and feature configuration
 */
FeaturesCollection& FeaturesCollection::train(const std::string& objectCategory, const std::string& classification, bool singleCell) {

    bool normal = true;
    if (classification == "Normal")
        normal = true;
    else if (classification == "Anomaly")
        normal = false;
    else
        throw std::invalid_argument("Feature Collections paramater classification should be \"Normal\" or \"Anomaly\"");
    

    for (auto& [feature, preProcessingPipeline] : features) {

        std::map<std::string, cv::Mat> images;
        readImagesFromDirectory(global::projectRoot + "data/" + objectCategory + "/Data/Images/" + classification + "/", images); 

        ObjectCoordinates objectBounds;
        for (auto& [imageName, image] : images) {

            preProcessingPipeline->apply(image, objectBounds);
            if (!normal) {
                std::cout << "Bounds: "
                          << objectBounds.xMin << ", "
                          << objectBounds.xMax << ", "
                          << objectBounds.yMin << ", "
                          << objectBounds.yMax << "\n";
                feature->updateFeature(image, singleCell, true, global::projectRoot + "data/" + objectCategory + "/Data/Images/" + classification + "/" + imageName, &objectBounds);
            }
            else
                feature->updateFeature(image, singleCell);

            break; //delete this
        }
    }

    return *this;
}

/*
 * train
 * DESCRIPTION
 * PARAMS
 */
FeaturesCollection& FeaturesCollection::train(cv::Mat& image, bool singleCell, std::string imageName) {

    cv::Mat imageClone = image.clone();

    if (imageName == "None")
        for (auto& [feature, preProcessingPipeline] : features) {

            image = imageClone.clone();
            preProcessingPipeline->apply(image);
            feature->updateFeature(image, singleCell, true, imageName);
        }
    else
        for (auto& [feature, preProcessingPipeline] : features) {

            image = imageClone.clone();
            preProcessingPipeline->apply(image);
            feature->updateFeature(image, singleCell);
        }

    return *this;
}

/*
 * extract
 * Returns all features as a mapping of feature name to matrix of feature values
 * @param normalFeatures - The mapping of feature names to feature matrix values
 */
FeaturesCollection& FeaturesCollection::extract(std::map<std::string, cv::Mat>& normalFeatures) {

    int index = 0;
    for (auto& [feature, preProcessingPipeline] : features) {
        index++;
            
        if (preProcessingPipeline->preProcessingConfiguration->mode == Mode::HSV) {

            for (int row = 0;  row < feature->featureMatrix.rows; row++) {
                for (int col = 0;  col < feature->featureMatrix.cols; col++) {
                    feature->featureMatrix.at<double>(row, col) = 100.0 - feature->featureMatrix.at<double>(row, col);
                }
            }
        }

        normalFeatures[feature->getName() + std::to_string(index)] = feature->featureMatrix;
    }

    return *this;
}

/*
 * reset
 * DESCRIPTION
 * PARAMS
 */
void FeaturesCollection::reset() {
    for (auto& [feature, preProcessingPipeline] : features)
        feature->reset();
}

/*
 * getFeatureNames
 * populates a list of feature namese as strings
 * @param featureNames - the vector of feature names to be populated
 */
void FeaturesCollection::getFeatureNames(std::vector<std::string>& featureNames) {
    for (const auto& [filter, pipeline] : features)
        featureNames.push_back(filter->getName());
}

std::ostream& operator<<(std::ostream& os, const FeaturesCollection& features) {
    os << "Features Configuration\n\n";

    

    for (const auto& [feature, preProcessingPipeline] : features.features) {
        if (typeid(*feature) == typeid(BinaryCountFeature))
            os << "Collecting Feature BinaryCountFeature: true\n";
        else if (typeid(*feature) == typeid(BinaryDistributionFeature))
            os << "Collecting Feature BinaryDistributionFeature: true\n";
    }

    return os;
}
