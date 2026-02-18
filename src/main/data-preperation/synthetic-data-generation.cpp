/*
 * synthetic-data-generation
 */

// OpenCV2
#include <opencv2/opencv.hpp>
// Fault Sense
#include "synthetic-data-generation.h"
#include "../general/file-operations/generic-file-operations.h"
#include "../general/generic-utils.h"



/*
 * generateRemoveNoiseTestData
 * Genereates synthetic images used for the testing of removeNoise function
 */
void generateRemoveNoiseTestData(cv::Mat &testImage) {

    // Check if data has already been generated

    /*
     * 2x2 center group
     * wwww
     * wbbw
     * wbbw
     * wwww
     */
    testImage = cv::Mat(4,4, CV_8UC1);
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {

            if (row == 0 || row == 3 || col == 3 || col == 0)
                testImage.at<uchar>(row, col) = 255;
            else
                testImage.at<uchar>(row, col) = 0;
        }
    }
}
