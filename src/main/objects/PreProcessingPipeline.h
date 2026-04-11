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
};
#endif
