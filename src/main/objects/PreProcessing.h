#ifndef PreProcessing_H
#define PreProcessing_H

// Standard
#include <exception>
#include <optional>
// OpenCV2
#include <opencv2/opencv.hpp>
// Fault Sense
#include "../pre-processing/object-detection.h"
#include "../pre-processing/pre-processing.h"
#include "../feature/feature-extraction.h"
#include "CannyThreshold.h"

enum class Mode { None, LBP, HSV, EDGE };
inline std::ostream& operator<<(std::ostream& os, const Mode& m) {
    switch (m) {
        case Mode::LBP:     os << "LBP";     break;
        case Mode::EDGE:    os << "EDGE"; break;
        case Mode::HSV:     os << "HSV"; break;
        default:            os << "None";  break;
    }
    return os;
}


/*
 * PreProcessing
 * A struct to  ...
 */
struct PreProcessing {

    Mode mode = Mode::None;
    bool applyObjectDetection = false; // Applies object detection on the new image and crops original image to the new bounds
    int noiseThreshold = 0; // if <= 0 no action taken i.e remove noise == false

    /*
     * apply
     * applies the configired pre-processing functions to the passed image
     *
     * @param image The image to which the specified pre-processing functions will be applied to
     */
    void apply(cv::Mat& image, std::optional<ObjectCoordinates> objectBounds = std::nullopt) const {

        cv::Mat originalImage = image.clone();
        if (applyObjectDetection && mode == Mode::LBP) 
            throw std::invalid_argument("Object detection input must be normalized to 0 || 255, i.e binary");


        if (mode == Mode::None) {
            return;
            
        } else if (mode == Mode::LBP) {
            lbpValues(image, originalImage);

        } else if (mode == Mode::EDGE) {
            CannyThreshold threshold{57, 29};
            cv::Mat kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
            edgeDetection(originalImage, kernal, threshold);

        } else if (mode == Mode::HSV) {
            HSV HSVThreshold{0, 22, 0, 119, 88,255};
            thresholdHSV(originalImage, HSVThreshold);
        }


        if (noiseThreshold > 0)
            removeNoise(originalImage, noiseThreshold);

        if (applyObjectDetection) {
            if (objectBounds.has_value()) {
                *objectBounds = getObject(originalImage);
            }
            objectDetection(originalImage, image);
        } else
            image = originalImage.clone();

    }

};
#endif
