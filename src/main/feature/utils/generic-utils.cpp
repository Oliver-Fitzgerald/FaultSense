/*
 * generic-utils
 * 
 */

// OpenCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Standard
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
// Fault Sense
#include "../objects/HSV.h"
#include "features.h"

void markFault(cv::Mat& image, int minX, int maxX, int minY, int maxY, const char* label);
void crop(cv::Mat& image, int minX, int maxX, int minY, int maxY, cv::Mat& returnImage);
void padImage(cv::Mat& image, int rows, int cols, cv::Mat& returnImage);
std::vector<cv::Mat> readImagesFromDirectory(const std::string& directory);


/*
 * markFault
 * given each edge point of a fault it draws a square to contain the fault and a label
 * to tag the square with
 */
void markFault(cv::Mat& image, int minX, int maxX, int minY, int maxY, const char* label) {

    cv::rectangle(image, cv::Point(minX - 10, minY - 10),cv::Point(maxX + 10, maxY + 10),cv::Scalar(0,0,255),3);
    putText(image, label, cv::Point(minX - 20, minY - 20), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,0,255), 2);
}

/*
 * crop
 *
 */
void crop(cv::Mat& image, int minX, int maxX, int minY, int maxY, cv::Mat& returnImage) {

    cv::Rect roi(minX - 10, minY - 10, (maxX - minX) + 20, (maxY - minY) + 20);
    returnImage = image(roi);
}

/*
 * padImage
 * Adds defined number of rows and cols of black pixels to the given image
 */
void padImage(cv::Mat& image, int rows, int cols, cv::Mat& returnImage) {

    cv::Mat temp;
    cv::Mat newRows = cv::Mat::zeros(cols, rows, CV_8UC1);
    cv::vconcat(image, newRows, temp);

    cv::Mat newCols = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::hconcat(temp, newCols, returnImage);
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
