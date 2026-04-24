#pragma once

// Standard
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <ostream>

// OpenCV
#include <opencv2/core.hpp>

// Forward declarations
class FeatureFilter;
class PreProcessingPipeline;

class FeaturesCollection {

public:
    std::map<std::unique_ptr<FeatureFilter>,
             std::unique_ptr<PreProcessingPipeline>> features;

    // Train using directory
    FeaturesCollection& train(const std::string& objectCategory,
                              const std::string& classification,
                              bool singleCell);

    // Train using single image
    FeaturesCollection& train(cv::Mat& image,
                              bool singleCell,
                              std::string imageName = "None");

    // Extract features
    FeaturesCollection& extract(std::map<std::string, cv::Mat>& normalFeatures);

    // Reset all features
    void reset();

    // Get feature names
    void getFeatureNames(std::vector<std::string>& featureNames);

    // Stream operator
    friend std::ostream& operator<<(std::ostream& os,
                                   const FeaturesCollection& features);
};
