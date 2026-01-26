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
#include <stdexcept>
#include <map>
// Fault Sense
#include "../objects/HSV.h"
#include "../objects/RGB.h"
#include "features.h"

void markFault(cv::Mat& image, int minX, int maxX, int minY, int maxY, const char* label, RGB colour);
void crop(cv::Mat& image, int minX, int maxX, int minY, int maxY, cv::Mat& returnImage);
void padImage(cv::Mat& image, int rows, int cols, cv::Mat& returnImage);
std::map<std::string, cv::Mat> readImagesFromDirectory(const std::string& directory);


/*
 * markFault
 * given each edge point of a fault it draws a square to contain the fault and a label
 * to tag the square with
 */
void markFault(cv::Mat& image, int minX, int maxX, int minY, int maxY, const char* label = nullptr, RGB colour  = RGB{255,0,0})
{
    cv::rectangle( image,
        cv::Point(minX, minY),
        cv::Point(maxX, maxY),
        cv::Scalar(colour.blue, colour.green, colour.red), 1
    );

    if (label && *label) {
        putText(
            image,
            label,
            cv::Point(minX - 20, minY - 20),
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            cv::Scalar(colour.blue, colour.green, colour.red),
            2
        );
    }
}

/*
 * crop
 *
 */
void crop(cv::Mat& image, int minX, int maxX, int minY, int maxY, cv::Mat& returnImage) {

    if (maxX <= minX) throw std::out_of_range("minX [" + std::to_string(minX) + "] must be less than maxX [" + std::to_string(maxX) + "]");
    if (maxY <= minY) throw std::out_of_range("minY [" + std::to_string(minY) + "] must be less than maxY [" + std::to_string(maxY) + "]");
    if (minX < 0) throw std::out_of_range("minX [" + std::to_string(minX) + "] must be greater than 0");
    if (minY < 0) throw std::out_of_range("minY [" + std::to_string(minY) + "] must be greater than 0");

    cv::Rect roi(minX, minY, (maxX - minX), (maxY - minY));
    returnImage = image(roi);
}

/*
 * padImage
 * Adds defined number of rows and cols of black pixels to the given image
 *
 * @param image Input image to pad
 * @param rows Number of rows to add (bottom padding)
 * @param cols Number of columns to add (right padding)
 * @param returnImage Output padded image
 */
void padImage(cv::Mat& image, int rows, int cols, cv::Mat& returnImage) {

    if (image.empty()) throw std::invalid_argument("Empty image cannot be padded");
    if (rows < 0 || cols < 0) throw std::invalid_argument("Padding of 0 cannot be applied\nrows: " + std::to_string(rows) + ", cols: " + std::to_string(cols));
    if (rows == 0 && cols == 0) ;

    cv::copyMakeBorder(image, returnImage, 
                       0, rows,           // top, bottom
                       0, cols,           // left, right
                       cv::BORDER_CONSTANT, 
                       cv::Scalar(0));    // black padding
}


/*
 * readImagesFromDirectory
 */
std::map<std::string, cv::Mat> readImagesFromDirectory(const std::string& directory) {

    namespace fs = std::filesystem;
    std::map<std::string, cv::Mat> images;
    
    try {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {

                std::string path = entry.path().string();
                std::string ext = entry.path().extension().string();
                
                // Check for common image extensions
                if (ext == ".JPG" || ext == ".jpeg" || ext == ".png" || 
                    ext == ".bmp" || ext == ".tiff" || ext == ".tif") {
                    
                    cv::Mat img = cv::imread(path);
                    if (!img.empty()) {
                        images.insert({path.substr(path.size() - 7), img});
                    } else {
                        std::cerr << "Failed to load: " << path << std::endl;
                    }
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
    
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
