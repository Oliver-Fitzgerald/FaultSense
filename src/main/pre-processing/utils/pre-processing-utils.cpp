/*
 * pre-processing-utils
 * This file contains supporting functions for pre-processing
 */

// OpenCV
#include <opencv2/opencv.hpp>
// Fault Sense
#include "pre-processing-utils.h"


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
 * binaryThreshold
 * Thresolds all pixel values in a grey scale image to 0 or 255
 * if values are less than 127 they are thresholded to 0 otherwise 255
 *
 * @param image The image to which thresholding will be applied
 * @param threshold An optional custom threeshold point 
 */
void binaryThreshold(cv::Mat& image, int threshold) {

    if (image.type() != CV_8UC1) throw std::invalid_argument("binaryThreshold may only be invoked on an image of type CV_8UC1");

    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            uchar* pixel = image.ptr<uchar>(row, col);

            if (*pixel < 127)
                *pixel = 0;
            else 
                *pixel = 255;
        }
    }
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

namespace pre_processing_utils {

    /*
     * mergeOverlappingGroups
     */
    bool mergeOverlappingGroups(PixelGroup &currentGroup, std::vector<PixelGroup> &pixelGroups, std::vector<bool> &grpUsed, int row) {

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
    void clean(PixelGroup &grp, cv::Mat &img, int minGrpSize) {

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
