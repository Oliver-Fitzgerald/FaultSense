/*
 * texture
 * Contains functions for analysis the texture of an image
 */

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Standard
#include <iostream>
#include <filesystem>
#include <bitset>
#include <cstdint>
#include <cmath>
// Fault Sense
#include "../pre-processing-utils.h"
#include "../generic-utils.h"
#include "../../object-detection.h"

uint8_t computePixelLBP(cv::Mat &image, int x, int y);
void computeLBP(cv::Mat &image, cv::Mat &LBPValues, float (&LBPHistogram)[5]);
double euclideanDistance(int x1, int y1, int x2, int y2);
cv::Mat brigthen_darker_areas(const cv::Mat& img, int threshold, int amount);

int amount = 0;
int threshold = 0;
int noise = 0;

/*
int main(int argc, char** argv) {

    cv::Mat originalImage = cv::imread("../../../../../../data/chewinggum/Data/Images/Anomaly/004.JPG");
    cv::Mat rawImage = cv::imread("../../../../../../data/chewinggum/Data/Images/Anomaly/004.JPG");
    //cv::Mat originalimage = cv::imread("../../../../../../data/sample-images/chewinggum-anomoly.jpg");
   // cv::Mat rawimage = cv::imread("../../../../../../data/sample-images/chewinggum-anomoly.jpg");
    cv::Mat LBPImage, croppedImage;

    // object detection
    HSV HSVThreshold{0, 22, 0, 119, 88,255}; thresholdHSV(rawImage, HSVThreshold);
    removeNoise(rawImage, 2000);
    objectCoordinates objectBounds = getObject(rawImage);
    crop(originalImage, objectBounds.yMin, objectBounds.yMax, objectBounds.xMin, objectBounds.xMax, croppedImage);


    // Show image
    cv::imshow("Image", LBPValues);
    cv::imshow("Image1",LBPImage);
    cv::imshow("Image2",finalImage);
    while (true) cv::pollKey();

    cv::namedWindow("Trackbars", (640,200));
    cv::createTrackbar("threshold", "Trackbars", &threshold, 255);
    cv::createTrackbar("amount", "Trackbars", &amount, 255);
    cv::createTrackbar("noise", "Trackbars", &noise, 1000000);
    int prev_threshold = threshold + 1;
    int prev_amount = amount + 1;
    int prev_noise = 0;
    cv::Mat exposed_img;

    cv::Mat temp, temp1;
    int paddingx = (3 - (croppedImage.cols % 3));
    int paddingy = (3 - (croppedImage.rows % 3));
    cv::cvtColor(croppedImage, temp, cv::COLOR_BGR2GRAY);
    padImage(temp, paddingy, paddingx, temp1);
    LBPImage = brigthen_darker_areas(temp1, threshold, amount);

    // Compute LBP for each 3x3 Cells
    cv::Mat LBPValues;
    int LBPHistogramWholeImage[5] = {0};
    computeLBP(LBPImage, LBPValues, LBPHistogramWholeImage);


    // Display Images
    while (true) {
        // Only reprocess if trackbar values changed
        if (threshold != prev_threshold || amount != prev_amount || noise != prev_noise) {
            std::cout << "Update\n";
            std::cout << "threshold: (" << prev_threshold << ", " << threshold << ")\n";
            std::cout << "amount: (" << prev_amount << ", " << amount << ")\n";
            std::cout << "noise: (" << prev_noise << ", " << noise << ")\n\n";


            // crop image to bounds divisible by 3
            cv::Mat temp, temp1;
            int paddingx = (3 - (croppedImage.cols % 3));
            int paddingy = (3 - (croppedImage.rows % 3));
            cv::cvtColor(croppedImage, temp, cv::COLOR_BGR2GRAY);
            padImage(temp, paddingy, paddingx, temp1);
            LBPImage = brigthen_darker_areas(temp1, threshold, amount);

            // Compute LBP for each 3x3 Cells
            cv::Mat LBPValues;
            int[5] LBPHistogram
            computeLBP(LBPImage, LBPValues, LBPHistogram);

            prev_threshold = threshold;
            prev_amount = amount;
            cv::imshow("Image", LBPValues);
        }
        
        int key = cv::waitKey(30); // Use waitKey instead of pollKey for better CPU usage
        if (key == 27 || key == 'q') break; // ESC or 'q' to exit
    }

}
*/

/*
 * computeLBP
 *
 * @param image The image for which Local Binary Pattern values are computed
 * @param LBPValues The image of Local Computed Binary values computed for 3x3 cells
 */
void computeLBP(cv::Mat &image, cv::Mat &LBPValues, float (&LBPHistogram)[5]) {

    // Compute for 3x3 Grid
    /*
    LBPValues = cv::Mat::zeros(image.rows / 3, image.cols / 3, CV_8UC1);
    for (int x = 1; x < image.rows; x+=3) {
        for (int y = 1; y < image.cols; y+=3) {
            LBPValues.at<uint8_t>((x - 1)/ 3, (y - 1) / 3) = computePixelLBP(image, x, y);
        }
    }
    */

    //Compute for every pixel
    // Form histogram of cell values in image
    LBPValues = cv::Mat::zeros(image.rows - 2, image.cols - 2, CV_8UC1);
    int itterations = 0;
    float total = 0;
    float weigth = 100 / ((float(image.cols - 2)) * (float(image.rows - 2)));
    for (int x = 1; x < image.rows - 1; x++) {
        for (int y = 1; y < image.cols - 1; y++) {
            itterations++;

            int value =  computePixelLBP(image, x, y);
            LBPValues.at<uint8_t>(x - 1 ,y - 1) = value;

            int index = std::clamp(value / 51, 0, 4);
            total += weigth;
            LBPHistogram[index] += weigth;
        }
    }

    /*
    std::cout << "=====================================\n";
    std::cout << "itterations: " << itterations << "\n";
    std::cout << "totalPixels: " << (image.cols - 2) * (image.rows - 2) << "\n";
    std::cout << "1 percent: " << weigth << "\n";
    std::cout << "total (100): " << total << "\n";
    std::cout << "=====================================\n";
    */
    
}

/*
 * computePixelLBP
 */
uint8_t computePixelLBP(cv::Mat &image, int x, int y) {

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

double euclideanDistance(int x1, int y1, int x2, int y2) {
    int dx = x2 - x1;
    int dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

cv::Mat brigthen_darker_areas(const cv::Mat& img, int threshold, int amount) {

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
