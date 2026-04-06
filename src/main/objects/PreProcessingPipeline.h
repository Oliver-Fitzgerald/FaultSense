#ifndef pre_processing_pipeline_H
#define pre_processing_pipeline_H

// Standard
#include <vector>
//Fault Sense
#include "PreProcessing.h"
// OpenCV2
#include <opencv2/opencv.hpp>

struct PreProcessingPipeline {

    std::vector<PreProcessing> steps;

    /*
     * apply
     * Applies each pre-processing step sequentially
     *@param image The image which each step will be applied to
     */
    void apply(cv::Mat &image) const {

        cv::Mat originalImage = image.clone();
        for (const PreProcessing &step : steps)
            step.apply(image);
    }
    /*
     * apply
     * Applies each pre-processing step sequentially
     *@param image The image which each step will be applied to
     */
    void apply(cv::Mat &image, ObjectCoordinates& objectBounds) const {

        for (const PreProcessing &step : steps)
            step.apply(image, &objectBounds);
    }
};
#endif
