/*
 * evaluation-utils
 */

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <map>
#include <string>
#include <array>
// FaultSense
#include "../../global-variables.h"
#include "../feature/feature-extraction.h"
#include "../objects/EvaluationMetrics.h"
#include "../objects/FeaturesCollection.h"
#include "../classification/ClassificationModel.h"

int euclidianDistance(std::array<float, 5>& pointOne, std::array<float, 5>& pointTwo);
int euclidianDistance(std::array<float, 5>& pointOne, float* pointTwo);
void initMatrix(const std::map<std::string, cv::Mat>::iterator &iterator, cv::Mat &categoryNorm);
void initMatrix(const cv::Mat &sampleImage, cv::Mat &categoryNorm);

namespace evaluate_utils {

    /*
     * evaluateImage
     * Determines wether an image is closer to given normal sample or a given anomally sample.
     * It is determined by a majority vote over each cell in the image.
     */
    bool evaluateImage(cv::Mat& image, FeaturesCollection& features, std::map<std::string, cv::Mat>& normalFeatures, std::map<std::string, cv::Mat>& anomalyFeatures, EvaluationMetrics& evaluationMetrics) {


        int rowMargin = image.rows % global::cellSize;
        int colMargin = image.cols % global::cellSize;

        int normalCellCount, anomalyCellCount = 0;

        std::unique_ptr<ClassificationModel> model = std::make_unique<SVM>(normalFeatures, anomalyFeatures);

        int collIndex, rowIndex = 0; 
        for (int row = rowMargin / 2; row + global::cellSize < image.rows - (rowMargin / 2); row += global::cellSize) {
            collIndex = 0;
            for (int col = colMargin / 2; col  + global::cellSize < image.cols - (colMargin / 2); col += global::cellSize) {

                // !!!!! LEFT OFF HERE !!!!!!!!!!!!!!!!!!!!!
                cv::Mat cell = image(cv::Range(row, row + global::cellSize), cv::Range(col, col + global::cellSize));
                bool normalClassification = model->classify(cell);

                if (normalClassification) {
                    normalCellCount++;
                    evaluationMetrics.averageNormalCells++;

                } else {
                    anomalyCellCount++; 
                    evaluationMetrics.averageAnomalyCells++;
                }

                cell.release();
                collIndex++;
            }
            rowIndex++;
        }

        std::cout << "normalCellCount:  " << normalCellCount << "\n";
        std::cout << "anomalyCellCount: " << anomalyCellCount << "\n\n";

        return model->result();
    }


    /*
     * whitePixelCount
     * Counts the number of white pixels in the passed image
     * @param image The image that the white pixels will be counted in
     */
    int countWhitePixels(cv::Mat &image) {

        int whitePixelCount = 0;
        for (int rows = 0; rows < image.rows; rows++) {

            for (int cols = 0; cols < image.cols; cols++) {
                
                int pixel = image.at<uchar>(rows, cols);
                if (pixel == 255)
                    whitePixelCount++;
            }
        }

        return whitePixelCount;
    }
}

int euclidianDistance(std::array<float, 5>& pointOne, std::array<float, 5>& pointTwo) {
    float sum = 0;

    for (int index = 0; index < 5; index++) {
        float diff = pointOne[index] - pointTwo[index];
        sum += diff * diff;
    }

    return sum;
}
int euclidianDistance(std::array<float, 5>& pointOne, float* pointTwo) {
    float sum = 0;

    for (int index = 0; index < 5; index++) {
        float diff = pointOne[index] - pointTwo[index];
        sum += diff * diff;
    }

    return sum;
}

/*
 * initMatrix
 * Initalizes a norm matrix to hold the distributions for each cell in an image of
 * dimensions sampleImage
 * @param sampleImage The iterator to an image whos dimensions are used to construct the norm matrix
 * @param categoryNorm The matrix norm initalized to reflect sampleImages dimensions
 */
void initMatrix(const std::map<std::string, cv::Mat>::iterator &iterator, cv::Mat &categoryNorm) {
    initMatrix(iterator->second, categoryNorm);
}

/*
 * initMatrix
 * Initalizes a norm matrix to hold the distributions for each cell in an image of
 * dimensions sampleImage
 * @param sampleImage The image dimensions used to construct the norm matrix
 * @param categoryNorm The matrix norm initalized to reflect sampleImages dimensions
 */
void initMatrix(const cv::Mat &sampleImage, cv::Mat &categoryNorm) {

    int rowMargin = sampleImage.rows % global::cellSize;
    int colMargin = sampleImage.cols % global::cellSize;

    categoryNorm = cv::Mat::zeros((sampleImage.rows - rowMargin) / global::cellSize, (sampleImage.cols - colMargin) / global::cellSize, CV_32FC(5));
}
