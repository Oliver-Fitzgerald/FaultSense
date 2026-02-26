/*
 * synthetic-data-generation
 */

// OpenCV2
#include <opencv2/opencv.hpp>
// Fault Sense
#include "synthetic-data-generation.h"
#include "../general/file-operations/generic-file-operations.h"
#include "../general/generic-utils.h"
// Standard
#include <array>


/*
 * generateRemoveNoiseTestData
 * Genereates synthetic images used for the testing of removeNoise function
 */
void generateRemoveNoiseTestData(std::array<cv::Mat, 4> &testImages) {

    cv::Mat testImage;

    /*
     * 2x2 center group
     * wwww
     * wbbw
     * wbbw
     * wwww
     */
    testImages[0] = cv::Mat(4,4, CV_8UC1);
    testImage = testImages[0];
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {

            if (row == 0 || row == 3 || col == 3 || col == 0)
                testImage.at<uchar>(row, col) = 0;
            else
                testImage.at<uchar>(row, col) = 255;
        }
    }

    // Image 001.JPG - V Shape
    // WWWWWWW
    // WBBWBBW
    // WWBBBWW
    // WWWWWWW
    testImages[1] = cv::Mat(4,7, CV_8UC1);
    testImage = testImages[1];
    for (int row = 0; row < testImage.rows; row++) {
        for (int col = 0; col < testImage.cols; col++) {

            // Edges White
            if (row == 0 || row == 3 || col == 6 || col == 0)
                testImage.at<uchar>(row, col) = 0;

            if (row == 1) {
                if (col == 0 || col == 3 || col == 6)
                    testImage.at<uchar>(row, col) = 0;
                else
                    testImage.at<uchar>(row, col) = 255;

            }
            if (row == 2) {
                if (col < 2 || col > 4)
                    testImage.at<uchar>(row, col) = 0;
                else
                    testImage.at<uchar>(row, col) = 255;

            }

        }
    }

    // Image 002.JPG - Inverted V Shape
    // WWWWWWW
    // WWBBBWW
    // WBBWBBW
    // WWWWWWW
    testImages[2] = cv::Mat(4,7, CV_8UC1);
    testImage = testImages[2];
    for (int row = 0; row < testImage.rows; row++) {
        for (int col = 0; col < testImage.cols; col++) {

            // Edges White
            if (row == 0 || row == 3 || col == 6 || col == 0)
                testImage.at<uchar>(row, col) = 0;

            if (row == 2) {
                if (col == 0 || col == 3 || col == 6)
                    testImage.at<uchar>(row, col) = 0;
                else
                    testImage.at<uchar>(row, col) = 255;

            }
            if (row == 1) {
                if (col < 2 || col > 4)
                    testImage.at<uchar>(row, col) = 0;
                else
                    testImage.at<uchar>(row, col) = 255;

            }

        }
    }

    // Image 003.JPG 
    // WWWWWWW
    // WBBBBBW
    // WBWWWBW
    // WBBBBBW
    // WWWWWWW
    testImages[3] = cv::Mat(5,7, CV_8UC1);
    testImage = testImages[3];
    for (int row = 0; row < testImage.rows; row++) {
        for (int col = 0; col < testImage.cols; col++) {

            // Edges White
            if (row == 0 || row == 4 || col == 6 || col == 0)
                testImage.at<uchar>(row, col) = 0;


            else if (row == 1 || row == 3) {
                if (col == 0 || col == 6 )
                    testImage.at<uchar>(row, col) = 0;
                else
                    testImage.at<uchar>(row, col) = 255;

            }
            else {
                if (col == 1 || col == 5)
                    testImage.at<uchar>(row, col) = 255;
                else
                    testImage.at<uchar>(row, col) = 0;

            }

        }
    }
}

/*
 * generateMergeOverlapTestData
 * Genereates synthetic images used for the testing of mergeOverlappingGroups function
 * from pre-processing-utils.cpp
 */
void generateRemoveNoiseTestData(std::array<cv::Mat, 5> &testImages) {

    cv::Mat testImage;

    // Image 000.png | Image 001.png | Image 002.png
    // WWWW          | WWWWW         | WWWWW
    // WBBW          | WBBWW         | WWBBW
    // WBBW          | WWBBW         | WBBWW
    // WWWW          | WWWWW         | WWWWW
    // Image 003.png | Image 004.png
    // WWWWW         | WWWWW
    // WWBWW         | WBBBW
    // WBBBW         | WWBWW
    // WWWWW         | WWWWW


    // Image 000.png
    testImages[0] = cv::Mat(4,4, CV_8UC1);
    testImage = testImages[0];
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {

            if (row == 0 || row == 3 || col == 3 || col == 0)
                testImage.at<uchar>(row, col) = 0;
            else
                testImage.at<uchar>(row, col) = 255;
        }
    }

    // Image 001.png
    // WWWWW        
    // WBBWW        
    // WWBBW        
    // WWWWW        
    testImages[1] = cv::Mat(4,5, CV_8UC1);
    testImage = testImages[1];
    for (int row = 0; row < testImage.rows; row++) {
        for (int col = 0; col < testImage.cols; col++) {

            // Edges White
            if (row == 0 || row == 3 || col == 4 || col == 0)
                testImage.at<uchar>(row, col) = 0;

            if (row == 1) {
                if (col == 0 || col == 3 || col == 4)
                    testImage.at<uchar>(row, col) = 0;
                else
                    testImage.at<uchar>(row, col) = 255;

            }
            if (row == 2) {
                if (col == 0 || col == 1 || col == 4)
                    testImage.at<uchar>(row, col) = 0;
                else
                    testImage.at<uchar>(row, col) = 255;

            }

        }
    }

    // Image 002.png
    // WWWWW
    // WWBBW
    // WBBWW
    // WWWWW
    testImages[2] = cv::Mat(4,5, CV_8UC1);
    testImage = testImages[2];
    for (int row = 0; row < testImage.rows; row++) {
        for (int col = 0; col < testImage.cols; col++) {

            // Edges White
            if (row == 0 || row == 3 || col == 4 || col == 0)
                testImage.at<uchar>(row, col) = 0;

            else if (row == 1) {
                if (col == 1)
                    testImage.at<uchar>(row, col) = 0;
                else
                    testImage.at<uchar>(row, col) = 255;

            }
            else if (row == 2) {
                if (col == 3)
                    testImage.at<uchar>(row, col) = 0;
                else
                    testImage.at<uchar>(row, col) = 255;

            }

        }
    }

    // Image 003.png
    // WWWWW
    // WWBWW
    // WBBBW
    // WWWWW
    testImages[3] = cv::Mat(4, 5, CV_8UC1);
    testImage = testImages[3];
    for (int row = 0; row < testImage.rows; row++) {
        for (int col = 0; col < testImage.cols; col++) {

            // Edges White
            if (row == 0 || row == 3 || col == 0 || col == 4)
                testImage.at<uchar>(row, col) = 0;


            else if (row == 1) {
                if (col < 2 || col > 2 )
                    testImage.at<uchar>(row, col) = 0;
                else
                    testImage.at<uchar>(row, col) = 255;

            }
            else {
                testImage.at<uchar>(row, col) = 255;
            }

        }
    }

    // Image 004.png
    // WWWWW
    // WBBBW
    // WWBWW
    // WWWWW
    testImages[4] = cv::Mat(4, 5, CV_8UC1);
    testImage = testImages[4];
    for (int row = 0; row < testImage.rows; row++) {
        for (int col = 0; col < testImage.cols; col++) {

            // Edges White
            if (row == 0 || row == 3 || col == 0 || col == 4)
                testImage.at<uchar>(row, col) = 0;


            else if (row == 2) {
                if (col < 2 || col > 2 )
                    testImage.at<uchar>(row, col) = 0;
                else
                    testImage.at<uchar>(row, col) = 255;

            }
            else if (row == 1) {
                testImage.at<uchar>(row, col) = 255;
            }

        }
    }
}
