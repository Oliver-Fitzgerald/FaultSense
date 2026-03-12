/*
 * feature-extraction
 * Extracts quantitative features from processed images
 */

// OpenCV
#include <opencv2/opencv.hpp>
// Standard
// Fault Sense

/*
 * lbpValueDistribution
 *
 * Calculate the distribution of pixel intensities accross the LBPValues Matrix
 * @param LBPValues (cv::Mat) The matrix of cumputed LBP Values
 * @param LBPHistogram (std::array<float, 5) The distribution of pixel intensities divided into 5 buckets
 */
void lbpValueDistribution(const cv::Mat &LBPValues, std::array<float, 5>& LBPHistogram) {

    // Calculate weigth of each pixel to normalize histogram from 0 - 100
    float weigth = 100 / ((float(LBPValues.cols)) * (float(LBPValues.rows)));

    for (int row = 0; row < LBPValues.rows; row++) {
        for (int col = 0; col < LBPValues.cols; col++) {

            int value = LBPValues.at<uint8_t>(row, col);
            int index = std::clamp(value / 51, 0, 4);
            LBPHistogram[index] += weigth;
        }
    }
}


