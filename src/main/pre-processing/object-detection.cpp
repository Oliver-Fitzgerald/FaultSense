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
#include "../objects/ObjectCoordinates.h"
#include "../general/generic-utils.h"
#include "utils/pre-processing-utils.h"

void objectDetection(cv::Mat &inputImage, cv::Mat &originalImage);
ObjectCoordinates getObject(cv::Mat &img);

/*
 * objectDetection
 */
void objectDetection(cv::Mat &inputImage, cv::Mat &originalImage) {

    ObjectCoordinates objectBounds = getObject(inputImage);
    inputImage = originalImage.clone();
    crop(inputImage, objectBounds.xMin, objectBounds.xMax, objectBounds.yMin, objectBounds.yMax, originalImage);
}

/*
 * objectDetection
 *
 * @param inputImage
 * @param returnImage
 */
void objectDetection(cv::Mat& inputImage, cv::Mat& originalImage, ObjectCoordinates& objectBounds) {
    crop(inputImage, objectBounds.xMin, objectBounds.xMax, objectBounds.yMin, objectBounds.yMax, originalImage);
}

/*
 * getObject
 */
ObjectCoordinates getObject(cv::Mat &img) {

    // Initalize with all set to max i.e image boundaries
    ObjectCoordinates coordinates{.xMin=img.rows,
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
    if (coordinates.xMin == img.rows) coordinates.xMin = 0;
    if (coordinates.xMax == 0) coordinates.xMax = img.rows;
    if (coordinates.yMin == img.cols) coordinates.yMin = 0;
    if (coordinates.yMax == 0) coordinates.yMax = img.cols;

    return coordinates;
}
