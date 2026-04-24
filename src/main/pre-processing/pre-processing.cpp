/*
 * pre-processing
 * Contains function for applying pre-processing techniques for the purposes
 * of object and feature extraction 
 * steps
 */

// OpenCV2
#include <opencv2/opencv.hpp>
// Fault Sense
#include "utils/pre-processing-utils.h"
#include "../objects/HSV.h"
#include "../objects/CannyThreshold.h"
#include "../objects/PixelCoordinates.h"

/*
 * lbpValues
 *
 * @param image The image for which Local Binary Pattern values are computed
 * @param LBPValues The image of Local Computed Binary values computed for 3x3 cells
 */
void lbpValues(const cv::Mat& image, cv::Mat& LBPValues) {

    pre_processing_utils::initMatrix(image, LBPValues);

    int rowMargin = (image.rows - 2) % 3;
    int colMargin = (image.cols - 2) % 3;

    int collIndex, rowIndex = 0; 
    for (int row = rowMargin / 2 + 1; row < image.rows - (rowMargin / 2) - 1; row += 3) {
        collIndex = 0;

        for (int col = colMargin / 2 + 1; col < image.cols - (colMargin / 2) - 1; col++) {

            int value = pre_processing_utils::pixelLBP(image, row, col);
            LBPValues.at<uint8_t>(rowIndex, collIndex) = value;

            collIndex++;
        }
        rowIndex++;
    }
}

/*
 * thresholdHSV
 * Applys a given color threshold to an image to highligth regions of an image
 */
void thresholdHSV(cv::Mat& image, HSV& threshold) {

    cv::Mat HSVImage;
    cv::cvtColor(image, HSVImage, cv::COLOR_BGR2HSV);
    cv::Scalar lower(threshold.hueMin, threshold.saturationMin, threshold.valueMin);
    cv::Scalar upper(threshold.hueMax, threshold.saturationMax, threshold.valueMax);
    cv::inRange(HSVImage, lower, upper, image);
}

/*
 * edgeDetection
 * applys edget detection an image and erodes the image with a given kernal
 */
void edgeDetection(cv::Mat& image, cv::Mat& kernal, CannyThreshold& threshold) {

    cv::Mat temp = image;
    cv::Canny(temp, image, threshold.lower, threshold.upper);
}


/*
 * removeNoise
 * removes groups of pixels in an image below the thresholded (minGrpSize) 
 * @param image
 * @param minGrpSize Groups smaller than minGrpSize are removed
 */
void removeNoise(cv::Mat& image, int minGrpSize) {
    using namespace std;

    bool lastPixel = false;
    std::vector<PixelGroup> pixelGroups;
    std::vector<bool> grpUsed;
    PixelGroup currentGroup = PixelGroup{.group = {},
                                          .bounds = {{-1, -1, -1}}};
    for (int x = 0; x < image.rows; x++) {


        for (int y = 0; y < image.cols; y++) {

            int pixel = image.at<uchar>(x, y);

            // Continue Current group
            if (pixel == 0 && lastPixel) {

                currentGroup.bounds[0].max = y - 1;
                bool existingGroup = pre_processing_utils::mergeOverlappingGroups(currentGroup, pixelGroups, grpUsed, x); // Move to function return bool (&grpUsed)

                // If it does not overlap with an existing group add as a new group
                if (!existingGroup) {

                    pixelGroups.push_back(currentGroup);
                    grpUsed.push_back(true);
                }

                currentGroup = PixelGroup{.group = {},
                                          .bounds = {{-1, -1, -1}}};
                lastPixel = false;



            // Start of new group
            } else if (pixel == 255) {

                currentGroup.group.push_back(PixelCoordinate{x,y});

                if (!lastPixel) {
                    lastPixel = true;
                    currentGroup.bounds[0].min = y;
                    currentGroup.row = x;
                }
            }

       }

        for (int i = 0; i < pixelGroups.size(); i++) {
            grpUsed[i] = pixelGroups[i].newRow(x);
        }

        // Remove any complete groups of size < minGrpSize
        for (int i = pixelGroups.size() - 1; i >= 0; i--) {

            if (!grpUsed[i] || !grpUsed[i] && (pixelGroups[i].group.size() < minGrpSize)) {

                pre_processing_utils::clean(pixelGroups[i],image, minGrpSize);
                grpUsed.erase(grpUsed.begin() + i);
                pixelGroups.erase(pixelGroups.begin() + i);
            }
        }
    }

    for (int i = pixelGroups.size() - 1; i >= 0; i--) {
        if (pixelGroups[i].group.size() < minGrpSize) {

            pre_processing_utils::clean(pixelGroups[i],image, minGrpSize);
            grpUsed.erase(grpUsed.begin() + i);
            pixelGroups.erase(pixelGroups.begin() + i);
        }
    }

    pixelGroups.clear();
    pixelGroups.shrink_to_fit();
    grpUsed.clear();
    grpUsed.shrink_to_fit();
}
