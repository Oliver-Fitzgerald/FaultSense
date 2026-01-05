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
#include "../objects/PixelCoordinates.h"
#include "features.h"

void thresholdHSV(cv::Mat& image, HSV& threshold);
void edgeDetection(cv::Mat& image, cv::Mat& kernal, CannyThreshold& threshold);
void removeNoise(cv::Mat& img);
void clean(pixelGroup &grp, cv::Mat &img);


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

    cv::Canny(image, image, threshold.lower, threshold.upper);
    cv::erode(image, image, kernal);
}


void removeNoise(cv::Mat& img) {
    using namespace std;

    int grpSize = 500;
    bool lastPixel = false;
    std::vector<pixelGroup> pixelGroups(grpSize);
    pixelGroup currentGroup = pixelGroup{.group = {},
                                         .min = -1,
                                         .max = -1,
                                         .redundant = false};

    for (int x = 0; x < img.rows; x++) {

        std::cout << "# Layer " << x << " / " << img.rows << ", Group Count: " << std::size(pixelGroups) << "\n";
        bool grpUsed[grpSize] = {false};

        for (int y = 0; y < img.cols; y++) {
            int pixel = img.at<uchar>(x, y);

            if (pixel == 0 && lastPixel) {

                currentGroup.max = y ;

                bool existingGroup = false;
                int lastHit = -1;

                for (int k = 0; k < size(pixelGroups); k++) {
                    int currentGroupLength = currentGroup.max - currentGroup.min;

                    if ((!pixelGroups[k].redundant) && (currentGroup.min >= pixelGroups[k].min && currentGroup.min <= pixelGroups[k].max) || (currentGroup.max >= pixelGroups[k].min && currentGroup.max <= pixelGroups[k].max) || (currentGroup.min < pixelGroups[k].min && currentGroup.min + currentGroupLength >= pixelGroups[k].min) || (currentGroup.max > pixelGroups[k].max && currentGroup.max - currentGroupLength <= pixelGroups[k].max) ) {
                        existingGroup = true;

                       if (lastHit > -1) {

                            pixelGroups[k].append(pixelGroups[lastHit]);
                            grpUsed[lastHit] = false;
                            lastHit = k;

                        } else {

                            pixelGroups[k].append(currentGroup);
                            grpUsed[k] = true;
                            lastHit = k;

                        }

                    } 

                } 
                if (!existingGroup) {

                    pixelGroups.push_back(currentGroup);
                    grpUsed[size(pixelGroups)] = true;
                    currentGroup = pixelGroup{.group = {},
                                              .min = -1,
                                              .max = -1,
                                              .redundant = false};

                } else
                    currentGroup = pixelGroup{.group = {},
                                              .min = -1,
                                              .max = -1,
                                              .redundant = false};


                lastPixel = false;


            } else if (pixel == 255) {
                currentGroup.group.push_back(pixelCoordinate{x,y});

                if (!lastPixel) {
                    lastPixel = true;
                    currentGroup.min = y - 1;
                }

            } else {
                lastPixel = false;
            }

        }


        std::vector<pixelGroup> tmp(200);
        for (int i = 0; i < size(pixelGroups); i++) {

            if (!grpUsed[i] || ( grpUsed[i] && pixelGroups[i].group.size() < 2000 && x + 1 == img.rows)) {
                clean(pixelGroups[i],img);

            } else if (grpUsed[i]) {
                std::cout << "group " << i << " size: " << size(pixelGroups[i].group) << "\n";
                tmp.push_back(pixelGroups[i]);
            }
        }
        pixelGroups = tmp;
    }

}

void clean(pixelGroup &grp, cv::Mat &img) {

    if (std::size(grp.group) < 2000)
        for (int j = 0; j < size(grp.group) ; j++) {

            /*
            std::cout << "          j: " << j << "\n";
            std::cout << "actual Size: " << std::size(grp.group) << "\n\n";
            */
            int row = grp.group[j].x;
            int col = grp.group[j].y;
            if (row >= 0 && row < img.rows && col >= 0 && col < img.cols) {
                img.at<uchar> vec(, );
            }
        }
    grp.group = {};

}

/*
 * main
 * for testing functionality
int main(int argc, char **argv) {

    std::string testImage = "../../../data/sample-images/board-scratch.JPG";
    cv::Mat image = cv::imread(testImage);
    cv::Mat markFaultImg = image, thresholdHSVImg = image, edgeDetection = image;

    cv::Mat kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
    markFault(markFaultImg, 400, 500, 330, 600, "Scratch");
    cv::imshow("Mark Fault", markFaultImg);

    HSV threshold{79, 179, 9, 52,10,255};
    thresholdHSV(thresholdHSVImg, threshold);
    cv::imshow("Mark Fault", thresholdHSVImg);

    edgeDetection(cv::Mat& image, cv::Mat& kernal);

    bool next = true;
    while (next) {

        int keyPressed = cv::pollKey();
        if (keyPressed == 'q')
            next = false;
    }
}
 */
