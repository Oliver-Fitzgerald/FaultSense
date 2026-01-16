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
#include "objects/PixelCoordinates.h"
#include "utils/generic-utils.h"
#include "utils/pre-processing-utils.h"

void objectDetection(cv::Mat &inputImage, cv::Mat &returnImage);
objectCoordinates getObject(cv::Mat &img);

/*
 * objectDetection
 */
void objectDetection(cv::Mat &inputImage, cv::Mat &returnImage) {

    HSV HSVThreshold{0, 22, 0, 119, 88,255}; thresholdHSV(inputImage, HSVThreshold);
    removeNoise(inputImage, 2000);
    objectCoordinates objectBounds = getObject(inputImage);
    crop(returnImage, objectBounds.yMin, objectBounds.yMax, objectBounds.xMin, objectBounds.xMax, returnImage);
}

/*
 * getObject
 */
objectCoordinates getObject(cv::Mat &img) {

    // Initalize with all set to max i.e image boundaries
    objectCoordinates coordinates{.xMin=img.rows,
                                  .xMax=0,
                                  .yMin=img.cols,
                                  .yMax=0};

    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            int pixel = img.at<uchar>(x, y);

            if (pixel == 255) {
                if (x < coordinates.xMin)
                    coordinates.xMin = x;
                else if (x > coordinates.xMax)
                    coordinates.xMax = x;

                if (y < coordinates.yMin)
                    coordinates.yMin = y;
                else if (y > coordinates.yMax)
                    coordinates.yMax = y;
            }
        }
    }

    return coordinates;
}
