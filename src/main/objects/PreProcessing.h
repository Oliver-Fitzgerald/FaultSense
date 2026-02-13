#ifndef pre_processing_object_H
#define pre_processing_object_H

// Standard
#include <exception>
// OpenCV2
#include <opencv2/opencv.hpp>
// Fault Sense
#include "../feature/object-detection.h"

/*
 * PreProcessing
 * A struct to  ...
 */
struct PreProcessing {

    bool lbp = false;
    bool hsv = false;
    bool edge = false;
    bool enableObjectDetection = false;
    int noiseThreshold = 0; // if <= 0 no action taken i.e remove noise == false

    /*
     * apply
     * applies the configired pre-processing functions to the passed image
     *
     * @param image The image to which the specified pre-processing functions will be applied to
     */
    void apply(cv::Mat &image) {

        cv::Mat originalImage = image.clone();

        if (enableObjectDetection && lbp) 
            throw std::invalid_argument("Object detection input must be normalized to 0 || 255, i.e binary");
        if ( (lbp && hsv) || (lbp && edge) || (edge && hsv) )
            throw std::invalid_argument("Only one of the following functions may be applied (\"hsv\", \"lbp\", \"edge\")");


        if (lbp) {
            lbpValues(image, originalImage);

        } else if (edge) {
            CannyThreshold threshold{57, 29};
            cv::Mat kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
            edgeDetection(originalImage, kernal, threshold);

        } else if (hsv) {
            HSV HSVThreshold{0, 22, 0, 119, 88,255};
            thresholdHSV(originalImage, HSVThreshold);
        }

        if (noiseThreshold > 0)
            removeNoise(originalImage, noiseThreshold);

        if (enableObjectDetection)
            objectDetection(originalImage, image);
        else
            image = originalImage.clone();

    }

};
#endif
