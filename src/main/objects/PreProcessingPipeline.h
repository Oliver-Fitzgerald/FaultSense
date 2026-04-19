#ifndef pre_processing_pipeline_H
#define pre_processing_pipeline_H

//Fault Sense
#include "PreProcessing.h"
// OpenCV2
#include <opencv2/opencv.hpp>

struct PreProcessingPipeline {

    std::optional<PreProcessing> objectDetectionConfiguration;
    std::optional<PreProcessing> preProcessingConfiguration;

    /*
     * apply
     * Applies each pre-processing step sequentially
     *@param image The image which each step will be applied to
     */
    void apply(cv::Mat &image) const {

        if (objectDetectionConfiguration.has_value())
            objectDetectionConfiguration->apply(image);

        if (preProcessingConfiguration.has_value())
            preProcessingConfiguration->apply(image);
    }

    /*
     * apply
     * Applies each pre-processing step sequentially
     *@param image The image which each step will be applied to
     */
    void apply(cv::Mat &image, ObjectCoordinates& objectBounds) const {

        if (objectDetectionConfiguration.has_value())
            objectDetectionConfiguration->apply(image, objectBounds);

        if (preProcessingConfiguration.has_value())
            preProcessingConfiguration->apply(image);
    }


    friend std::ostream& operator<<(std::ostream& os, const PreProcessingPipeline& p) {

        os << "Pre-Processing and Object Detection Configuration:\n\n";

        os << "Object Detection Applied:         " << (p.objectDetectionConfiguration.has_value() ? "true" : "false") << "\n";
        if (p.objectDetectionConfiguration.has_value()) {
            os << "Object Detection Mode:            " << p.objectDetectionConfiguration->mode << "\n";
            os << "Object Detection Noise Threshold: " << p.objectDetectionConfiguration->noiseThreshold << "\n\n";
        }

        os << "Pre Procesing Mode:               " << p.preProcessingConfiguration->mode << "\n";
        os << "Pre Processing Noise Threshold:   " << p.preProcessingConfiguration->noiseThreshold << "\n\n";

        return os;
    }
};
#endif
