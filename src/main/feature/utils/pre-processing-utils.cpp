/*
 * features
 * This file contains functions for extracting features from images
 */

// OpenCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Standard
#include <iostream>
// Fault Sense
#include "features.h"
#include "../../general/generic-utils.h"
#include "../../objects/HSV.h"
#include "../../objects/CannyThreshold.h"
#include "../../objects/PixelCoordinates.h"
#include "pre-processing-utils_internal.h"


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
    std::vector<pixelGroup> pixelGroups;
    std::vector<bool> grpUsed;
    pixelGroup currentGroup = pixelGroup{.group = {},
                                          .bounds = {{-1, -1, -1}}};
    for (int x = 0; x < image.rows; x++) {


        for (int y = 0; y < image.cols; y++) {

            int pixel = image.at<uchar>(x, y);

            // Continue Current group
            if (pixel == 0 && lastPixel) {

                currentGroup.bounds[0].max = y - 1;
                bool existingGroup = internal::mergeOverlappingGroups(currentGroup, pixelGroups, grpUsed, x); // Move to function return bool (&grpUsed)

                // If it does not overlap with an existing group add as a new group
                if (!existingGroup) {

                    pixelGroups.push_back(currentGroup);
                    grpUsed.push_back(true);
                }

                currentGroup = pixelGroup{.group = {},
                                          .bounds = {{-1, -1, -1}}};
                lastPixel = false;



            // Start of new group
            } else if (pixel == 255) {

                currentGroup.group.push_back(pixelCoordinate{x,y});

                if (!lastPixel) {
                    lastPixel = true;
                    currentGroup.bounds[0].min = y;
                    currentGroup.row = x;
                }
            }


           /* DEBUG INFO
            std::cout << "\n(row, col) => (" << x << ", " << y << ")\n";
            std::cout << "(image.rows, image.cols) => (" << image.rows << ", " << image.cols << ")\n";
            std::cout << "currentGroup.group.size() => " << currentGroup.group.size() << "\n";
            std::cout << "pixelGroups.size(): " << pixelGroups.size() << "\n";
            std::cout << "grpUsed.size(): " << grpUsed.size() << "\n";
            std::cout << "pixel: " <<  pixel << "\n";
            */

            //std::cout << "memory usage end row(" << x << "): " << getMemoryUsage() << "\n";
        }

        // Remove any complete groups of size < minGrpSize
        for (int i = pixelGroups.size() - 1; i >= 0; i--) {

            if (!grpUsed[i] || !grpUsed[i] && (pixelGroups[i].group.size() < minGrpSize)) {

                internal::clean(pixelGroups[i],image, minGrpSize);
                grpUsed.erase(grpUsed.begin() + i);
                pixelGroups.erase(pixelGroups.begin() + i);
            }
        }
    }

    for (int i = pixelGroups.size() - 1; i >= 0; i--) {
        if (pixelGroups[i].group.size() < minGrpSize) {

            internal::clean(pixelGroups[i],image, minGrpSize);
            grpUsed.erase(grpUsed.begin() + i);
            pixelGroups.erase(pixelGroups.begin() + i);
        }
    }

    pixelGroups.clear();
    pixelGroups.shrink_to_fit();
    grpUsed.clear();
    grpUsed.shrink_to_fit();
}

/*
 * illuminationInvariance
 */
void illuminationInvariance(const cv::Mat &image, cv::Mat &returnImage) {

    // Applying illumination invariance
    cv::Mat temp;
    cv::cvtColor(image, temp, cv::COLOR_BGR2GRAY);
    returnImage = internal::brigthenDarkerAreas(temp, 169, 46);
}

namespace internal {

    /*
     * brigthenDarkerAreas
     */
    cv::Mat brigthenDarkerAreas(const cv::Mat& img, const int threshold, const int amount) {

        cv::Mat returnImage = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);

        for (int row = 0; row < img.rows; row++) {
            for (int col = 0; col < img.cols; col++) {

                int pixel = img.at<uint8_t>(row, col);
                if (pixel < threshold)
                    returnImage.at<uint8_t>(row,col) = pixel + amount;
                else
                    returnImage.at<uint8_t>(row,col) = pixel;
            }
        }
        
        return returnImage;
    }

    /*
     * mergeOverlappingGroups
     */
    bool mergeOverlappingGroups(pixelGroup &currentGroup, std::vector<pixelGroup> &pixelGroups, std::vector<bool> &grpUsed, int row) {

        int prevOveralapIndex = -1;
        bool existingGroup = false;

        // Add to any overlapping group
        for (int k = 0; k < std::size(pixelGroups); k++) {

            //std::cout << "Memory usage before group " << k << ": " << getMemoryUsage() << "\n";
            // std::cout << "pixelGrpup[" << k << "] bounds size: " <<  pixelGroups[k].bounds.size() << "\n";
            // std::cout << "currentGroup[" << k << "] bounds size: " <<  currentGroup.bounds.size() << "\n";
            //
            // std::cout << "pixelGrpup[" << k << "] group size: " <<  pixelGroups[k].group.size() << "\n";
            // for each sub-group (min, max)
            for (int index = 0; index < currentGroup.bounds.size(); index++) {
            for (int pixelGroupIndex = 0; pixelGroupIndex < pixelGroups[k].bounds.size(); pixelGroupIndex++) {

                int currentGroupLength = currentGroup.bounds[index].max - currentGroup.bounds[index].min;

                if ( // Check if the current group overlaps with an existing group
                    (currentGroup.bounds[index].min >= pixelGroups[k].bounds[pixelGroupIndex].min && currentGroup.bounds[index].min <= pixelGroups[k].bounds[pixelGroupIndex].max) || 
                    (currentGroup.bounds[index].max >= pixelGroups[k].bounds[pixelGroupIndex].min && currentGroup.bounds[index].max <= pixelGroups[k].bounds[pixelGroupIndex].max) || 
                    (currentGroup.bounds[index].min < pixelGroups[k].bounds[pixelGroupIndex].min && currentGroup.bounds[index].min + currentGroupLength >= pixelGroups[k].bounds[pixelGroupIndex].min) || 
                    (currentGroup.bounds[index].max > pixelGroups[k].bounds[pixelGroupIndex].max && currentGroup.bounds[index].max - currentGroupLength <= pixelGroups[k].bounds[pixelGroupIndex].max) 
                   ) {

                    existingGroup = true;

                    if (prevOveralapIndex > -1) {

                        pixelGroups[k].append(pixelGroups[prevOveralapIndex], row);
                        grpUsed[prevOveralapIndex] = false;
                        prevOveralapIndex = k;

                    } else {

                        pixelGroups[k].append(currentGroup, row);
                        grpUsed[k] = true;
                        prevOveralapIndex = k;

                    }
                } 

            }
            }
            // std::cout << "Memory usage after group " << k << ": " << getMemoryUsage() << "\n";
        }
        return existingGroup;
    }

    /*
     * clean
     * Sets all of the pixel co-ordinates in a group of pixels to 0 i.e removes them.
     * From an image
     * @param grp
     * @param img
     * @param minGrpSize
     */
    void clean(pixelGroup &grp, cv::Mat &img, int minGrpSize) {

        if (std::size(grp.group) < minGrpSize)
            for (int j = 0; j < size(grp.group) ; j++) {

                /*
                std::cout << "          j: " << j << "\n";
                std::cout << "actual Size: " << std::size(grp.group) << "\n\n";
                */
                int row = grp.group[j].x;
                int col = grp.group[j].y;
                if (row >= 0 && row < img.rows && col >= 0 && col < img.cols) {
                    img.at<uchar>(row, col) = 0;
                }
            }
        grp.group = {};
    }
}
