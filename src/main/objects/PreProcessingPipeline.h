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

        for (const PreProcessing &step : steps)
            step.apply(image);
    }
};
#endif
