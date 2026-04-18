/*
 * synthethic-features-data-generation
 * Generates the synthetic test data used in Features-tests unit tests
 */

// OpenCV2
#include <opencv2/opencv.hpp>

/*
 * generateFeatureTestData
 * Generates test images (opencv matrixs) with data for testing 
 * @param images The matrixs to be populated with test data
 */
void generateFeatureTestData(std::vector<cv::Mat>& images) {

    cv::Mat image;

    /* 2x2 center group with values at 0 | 255 */
    image = cv::Mat(4,4, CV_8UC1);
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {

            if (row == 0 || row == 3 || col == 3 || col == 0)
                image.at<uchar>(row, col) = 0;
            else
                image.at<uchar>(row, col) = 255;
        }
    }
    images.push_back(image);

    /* 2x2 center group with values ranging from 0 -> 255 */
    image = cv::Mat(4,4, CV_8UC1);
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {

            if (row == 0 || row == 3 || col == 3 || col == 0)
                image.at<uchar>(row, col) = 0;
            else
                image.at<uchar>(row, col) = 255;
        }
    }
    images.push_back(image);
}
