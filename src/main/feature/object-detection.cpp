/*
 * object detection
 * seperates an image into foreground and background
 * Note: Currently only expects one object within the image
 *
 * Abbreviations:
 * cols = columns
 */

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Fault Sense
#include "../objects/PixelCoordinates.h"
#include "utils/generic-utils.h"
#include "utils/pre-processing-utils.h"

void getCategory(const std::string& imageCategory, std::map<std::string, bool>& category);
void getObject(cv::Mat &img, ObjectCoordinates& objectBounds);

/*
 * objectDetection
 */
cv::Mat objectDetection(cv::Mat &inputImage, cv::Mat &returnImage, const std::string& imageCategory, ObjectCoordinates& objectBounds) {

    std::map<std::string, bool> category = {
        {"chewinggum", false},
        {"cashew", false}
    };
    getCategory(imageCategory, category);
    cv::Mat image = inputImage.clone();

    if (category["chewinggum"]) {
        HSV HSVThreshold{0, 22, 0, 119, 88,255}; thresholdHSV(image, HSVThreshold);
    } else if (category["cashew"]) {
        HSV HSVThreshold{8, 179, 57, 255, 164,255}; thresholdHSV(image, HSVThreshold);
    } else 
        throw std::invalid_argument("object-detecton not configured for objectCategory: " + imageCategory);


    //while (cv::pollKey() != 113) cv::imshow("Image1", image);
    removeNoise(image, 1000);

    getObject(image, objectBounds);
    crop(inputImage, objectBounds.xMin, objectBounds.xMax, objectBounds.yMin, objectBounds.yMax, returnImage);
    return image;
}

/*
 * getObject
 */
void getObject(cv::Mat &img, ObjectCoordinates& objectBounds) {

    // Initalize with all set to max i.e image boundaries
    objectBounds = ObjectCoordinates{.xMin=img.rows,
                                  .xMax=0,
                                  .yMin=img.cols,
                                  .yMax=0};

    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            int pixel = img.at<uchar>(x, y);

            if (pixel == 255) {
                if (x < objectBounds.xMin)
                    objectBounds.xMin = x;
                else if (x > objectBounds.xMax)
                    objectBounds.xMax = x;

                if (y < objectBounds.yMin)
                    objectBounds.yMin = y;
                else if (y > objectBounds.yMax)
                    objectBounds.yMax = y;
            }
        }
    }
}

    
void getCategory(const std::string& imageCategory, std::map<std::string, bool>& category) {

    for (auto& [objectCategory, apply] :  category ) {
        if (imageCategory == objectCategory)
            apply = true;
    }
}

