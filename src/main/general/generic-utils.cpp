/*
 * generic-utils
 * Contains functions that have generalt utility throughout the project but
 * cannot be directly categorized into a specific module. 
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
#include <sys/resource.h>
// Fault Sense
#include "../objects/HSV.h"
#include "../objects/RGB.h"
#include "features.h"

namespace keys {
    const int POSITIVE = 43;
    const int NEGATIVE = 95;
}


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
 * THIS FUNCTION PROBABLY HAS VARRIABLE NAMES MIXED UP I.E ROWS AND COLS MAY NEED TO RENAME
 */
void crop(cv::Mat& image, int minX, int maxX, int minY, int maxY, cv::Mat& returnImage) {

    if (maxX <= minX) throw std::out_of_range("minX [" + std::to_string(minX) + "] must be less than maxX [" + std::to_string(maxX) + "]");
    if (maxY <= minY) throw std::out_of_range("minY [" + std::to_string(minY) + "] must be less than maxY [" + std::to_string(maxY) + "]");
    if (minX < 0) throw std::out_of_range("minX [" + std::to_string(minX) + "] must be greater than 0");
    if (minY < 0) throw std::out_of_range("minY [" + std::to_string(minY) + "] must be greater than 0");
    if (minX + (maxX - minX) > image.rows) throw std::out_of_range("maxX [" + std::to_string(maxX) + "] must be less than image.rows: " + std::to_string(image.rows));
    if (minY + (maxY - minY) > image.cols) throw std::out_of_range("maxY [" + std::to_string(maxY) + "] must be less than image.cols: " + std::to_string(image.cols));

    /* DEBUG INFO
    std::cout << "\n\nimage.rows: " << image.rows << "\n";
    std::cout << "image.cols: " << image.cols << "\n";
    std::cout << "maxX: " << maxX << "\n";
    std::cout << "maxY: " << maxY << "\n\n";

    std::cout << "minX: " << minX << "\n";
    std::cout << "maxX - minX: " << maxX - minX << "\n";
    std::cout << "minY: " << minY << "\n";
    std::cout << "maxY - minY: " << maxY - minY << "\n";
    */


    cv::Rect roi(minY, minX, (maxY - minY), (maxX - minX));
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
 * getMemoryUsage
 */
long getMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // kB on Linux, bytes on macOS
}


/*
 * zoom
 * Zooms in or out on an image
 *
 * @parm key the key pressed 
 * @param image the image to be zoomed in or out of
 * @param retult the image zoomed in or out
 */
double zoomFactor = 1.0;
int zoom(int key, cv::Mat &image, cv::Mat &result) {

    if (key == keys::POSITIVE) {
        zoomFactor *= 1.5;
    } else if (key == keys::NEGATIVE) {
        zoomFactor *= 0.5;
    } else 
        return key;

    cv::resize(image, result, cv::Size(), zoomFactor, zoomFactor, cv::INTER_NEAREST);
    return key;
}
