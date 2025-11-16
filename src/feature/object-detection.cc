/*
 * object detection
 * seperates an image into foreground and background
 * Note: Only expects one object within the image
 *
 * Abbreviations:
 * cols = columns
 */

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Standard
#include <iostream>
#include <filesystem>
#include <vector>
// Fault Sense
#include "features.h"

struct pixelCoordinate {
    int x;
    int y;
};

struct pixelGroup {
    std::vector<pixelCoordinate> group;
    int min;
    int max;
    bool redundant;

    void append(pixelGroup& theOtherGroup) {
        // Does not account for the case of a fork
        max = theOtherGroup.max;
        min = theOtherGroup.min;

        group.insert(group.end(), 
             theOtherGroup.group.begin(), 
             theOtherGroup.group.begin() + std::size(theOtherGroup.group));
    }
};

struct objectCoordinates {
    int xMin;
    int xMax;
    int yMin;
    int yMax;
};


void applyCanny(cv::Mat& image, cv::Mat& returnImage);
void removeNoise(cv::Mat& img);
void itterateCols();
void clean(pixelGroup &grp, cv::Mat &img);
objectCoordinates getEdges(cv::Mat &img);
std::vector<cv::Mat> readImagesFromDirectory(const std::string& directory);

int main(int argc, char** argv) {

    
    
    std::vector<cv::Mat> images = readImagesFromDirectory("../data/chewinggum/Data/Images/Anomaly/");
    std::vector<cv::Mat> finalImages;

    std::cout << std::size(images) << "\n";
    for (int i = 0; i < std::size(images); i++) {

        std::cout << "Object " << i + 1 << " Detection: ";
        cv::Mat img,image = images[i];
        applyCanny(image, img);
        std::cout << ".";
        removeNoise(img);
        objectCoordinates coordinates = getEdges(img);
        markFault(image, coordinates.yMin, coordinates.yMax, coordinates.xMin, coordinates.xMax, "Object");
        std::cout << ".\n";

        finalImages.push_back(image);
        finalImages.push_back(img);
    }
        
    std::cout << "Object Detection Complete\n";

    // Display each image in a separate window
    for (int i = 0; i < finalImages.size(); i++) {
        std::string windowName;
        
        windowName = "Image " + std::to_string(i + 1);
        
        // Create and show window
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        cv::imshow(windowName, finalImages[i]);
        
        // Position windows in a cascading pattern (optional)
        //moveWindow(windowName, 50 + (i * 50), 50 + (i * 50));
    }
    
    std::cout << "Displaying " << images.size() << " images in separate windows." << std::endl;
    std::cout << "Press any key to close all windows..." << std::endl;
    while (true) cv::pollKey();
}

void applyCanny(cv::Mat& image, cv::Mat& returnImage){
    cv::Mat temp,temp1,temp2,temp3,temp4;

    cv::cvtColor(image, temp, cv::COLOR_BGR2GRAY);
    cv::Canny(temp, temp1, 215, 108);
    cv::Mat dilateKernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::dilate(temp1, returnImage, dilateKernal);
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
                img.at<uchar>(row, col) = 0;
            }
        }
    grp.group = {};

}

objectCoordinates getEdges(cv::Mat &img) {

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

std::vector<cv::Mat> readImagesFromDirectory(const std::string& directory) {

    namespace fs = std::filesystem;
    std::vector<cv::Mat> images;
    
    std::cout << "Reading files ";
    try {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {

                std::cout << ".";
                std::string path = entry.path().string();
                std::string ext = entry.path().extension().string();
                
                // Check for common image extensions
                if (ext == ".JPG" || ext == ".jpeg" || ext == ".png" || 
                    ext == ".bmp" || ext == ".tiff" || ext == ".tif") {
                    
                    cv::Mat img = cv::imread(path);
                    if (!img.empty()) {
                        images.push_back(img);
                        std::cout << "Loaded: " << path << std::endl;
                    } else {
                        std::cerr << "Failed to load: " << path << std::endl;
                    }
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
    std::cout << "\n";
    
    return images;
}
