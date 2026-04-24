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
#include "../objects/HSV.h"
#include "../objects/CannyThreshold.h"
#include "../../objects/PixelCoordinates.h"
#include "features.h"

void thresholdHSV(cv::Mat& image, HSV& threshold);
void edgeDetection(cv::Mat& image, cv::Mat& kernal, CannyThreshold& threshold);
void removeNoise(cv::Mat& img, int minGrpSize);
void clean(pixelGroup &grp, cv::Mat &img, int minGrpSize);
void illuminationInvariance(const cv::Mat &image, cv::Mat &returnImage);
cv::Mat brigthenDarkerAreas(const cv::Mat& img, const int threshold, const int amount);
bool mergeOverlappingGroups(pixelGroup &currentGroup, std::vector<pixelGroup> &pixelGroups, std::vector<bool> &grpUsed);

namespace pre_processing_utils {

    bool mergeOverlappingGroups(pixelGroup &currentGroup, std::vector<pixelGroup> &pixelGroups, std::vector<bool> &grpUsed, int row);
    void clean(pixelGroup &grp, cv::Mat &img, int minGrpSize);
    void initMatrix(const cv::Mat &sampleImage, cv::Mat &categoryNorm);
    uint8_t pixelLBP(const cv::Mat &image, const int x, const int y);
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
                bool existingGroup = pre_processing_utils::mergeOverlappingGroups(currentGroup, pixelGroups, grpUsed, x); // Move to function return bool (&grpUsed)
                // ^ mergeOverlappingGroups

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

       }

        for (int i = 0; i < pixelGroups.size(); i++) {
            grpUsed[i] = pixelGroups[i].newRow(x);
        }

        // Remove any complete groups of size < minGrpSize
        for (int i = pixelGroups.size() - 1; i >= 0; i--) {

            if (!grpUsed[i] || !grpUsed[i] && (pixelGroups[i].group.size() < minGrpSize)) {

                pre_processing_utils::clean(pixelGroups[i],image, minGrpSize);
                // ^ clean
                grpUsed.erase(grpUsed.begin() + i);
                pixelGroups.erase(pixelGroups.begin() + i);
            }
        }
    }

    for (int i = pixelGroups.size() - 1; i >= 0; i--) {
        if (pixelGroups[i].group.size() < minGrpSize) {

            pre_processing_utils::clean(pixelGroups[i],image, minGrpSize);
            // ^ clean
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

/*
 * illuminationInvariance
 */
void illuminationInvariance(const cv::Mat &image, cv::Mat &returnImage) {

    // Applying illumination invariance
    cv::Mat temp;
    cv::cvtColor(image, temp, cv::COLOR_BGR2GRAY);
    returnImage = brigthenDarkerAreas(temp, 169, 46);
}

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
bool mergeOverlappingGroups(pixelGroup &currentGroup, std::vector<pixelGroup> &pixelGroups, std::vector<bool> &grpUsed) {

    int prevOveralapIndex = -1;
    bool existingGroup = false;

    // Add to any overlapping group
    for (int k = 0; k < size(pixelGroups); k++) {
        int currentGroupLength = currentGroup.max - currentGroup.min;

        if ( // Check if the current group overlaps with an existing group
            (currentGroup.min >= pixelGroups[k].min && currentGroup.min <= pixelGroups[k].max) || 
            (currentGroup.max >= pixelGroups[k].min && currentGroup.max <= pixelGroups[k].max) || 
            (currentGroup.min < pixelGroups[k].min && currentGroup.min + currentGroupLength >= pixelGroups[k].min) || 
            (currentGroup.max > pixelGroups[k].max && currentGroup.max - currentGroupLength <= pixelGroups[k].max) 
           ) {

            existingGroup = true;

            if (prevOveralapIndex > -1) {

                pixelGroups[k].append(pixelGroups[prevOveralapIndex], false);
                grpUsed[prevOveralapIndex] = false;
                prevOveralapIndex = k;

            } else {

                pixelGroups[k].append(currentGroup, true);
                grpUsed[k] = true;
                prevOveralapIndex = k;

            }

        } 

    }
    return existingGroup;
}
 */
namespace pre_processing_utils {

    /*
     * mergeOverlappingGroups
     */
    bool mergeOverlappingGroups(pixelGroup &currentGroup, std::vector<pixelGroup> &pixelGroups, std::vector<bool> &grpUsed, int row) {

        int prevOveralapIndex = -1;
        bool existingGroup = false;

        // Add to any overlapping group
        for (int k = 0; k < std::size(pixelGroups); k++) {

            for (int index = 0; index < currentGroup.bounds.size(); index++) {
            for (int pixelGroupIndex = 0; pixelGroupIndex < pixelGroups[k].bounds.size(); pixelGroupIndex++) {

                int currentGroupLength = currentGroup.bounds[index].max - currentGroup.bounds[index].min;

                if ( // Check if the current group overlaps with an existing group
                    (currentGroup.bounds[index].min - 1 >= pixelGroups[k].bounds[pixelGroupIndex].min - 1 && currentGroup.bounds[index].min - 1 <= pixelGroups[k].bounds[pixelGroupIndex].max + 1) || 
                    (currentGroup.bounds[index].max + 1 >= pixelGroups[k].bounds[pixelGroupIndex].min - 1 && currentGroup.bounds[index].max + 1 <= pixelGroups[k].bounds[pixelGroupIndex].max + 1) || 
                    (currentGroup.bounds[index].min - 1 < pixelGroups[k].bounds[pixelGroupIndex].min - 1 && currentGroup.bounds[index].min - 1 + currentGroupLength >= pixelGroups[k].bounds[pixelGroupIndex].min - 1) || 
                    (currentGroup.bounds[index].max + 1 > pixelGroups[k].bounds[pixelGroupIndex].max + 1 && currentGroup.bounds[index].max + 1 - currentGroupLength <= pixelGroups[k].bounds[pixelGroupIndex].max + 1) 
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
            for (int j = 0; j < grp.group.size() ; j++) {

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

    void initMatrix(const cv::Mat &sampleImage, cv::Mat &categoryNorm) {

        int rowMargin = (sampleImage.rows - 2) % 3;
        int colMargin = (sampleImage.cols - 2) % 3;
        int rows = (sampleImage.rows - 2 - rowMargin) / 3;
        int cols = (sampleImage.cols - 2 - colMargin);
        categoryNorm = cv::Mat::zeros(rows, cols, CV_8UC1);
    }

    /*
     * pixelLBP
     */
    uint8_t pixelLBP(const cv::Mat &image, const int x, const int y) {

        uint8_t LBDValue = 0b00000000;
        int centerValue = image.at<uchar>(x,y);
        
        if (image.at<uchar>(x - 1, y - 1) >= centerValue) 
            LBDValue |= (1 << 0);
        if (image.at<uchar>(x - 1, y) >= centerValue) 
            LBDValue |= (1 << 1);
        if (image.at<uchar>(x - 1, y + 1) >= centerValue) 
            LBDValue |= (1 << 2);

        // Inline
        if (image.at<uchar>(x, y - 1) >= centerValue) 
            LBDValue |= (1 << 3);
        if (image.at<uchar>(x, y + 1) >= centerValue) 
            LBDValue |= (1 << 4);

        // Below
        if (image.at<uchar>(x + 1, y - 1) >= centerValue) 
            LBDValue |= (1 << 5);
        if (image.at<uchar>(x + 1, y) >= centerValue) 
            LBDValue |= (1 << 6);
        if (image.at<uchar>(x + 1, y + 1) >= centerValue) 
            LBDValue |= (1 << 7);
        
        return LBDValue;
    }

}
