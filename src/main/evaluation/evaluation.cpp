/*
 * evaluation
 * functions for the purpose of evaluating the effectivness of trained 
 * models/norms at classifing a samples
 */

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <map>
#include <array>
#include <string>
// Fault Sense
#include "evaluation_internal.h"
#include "../general/generic-utils.h"
#include "../feature/feature-extraction.h"
#include "../feature/pre-processing.h"
#include "../feature/object-detection.h"
#include "../objects/PreProcessingPipeline.h"


/*
 * evaluateObjectCategory
 * Given an object category it iterates over all normal and anomoly sample from the given category and prints 
 * the evaluation of each image as well as the total and average evaluations for normal and anomaly samples
 * @param objectCategory
 * @param normalMatrixNorm
 * @param anomalyDistributionNorm
 * @param preProcessingPipeline
 */
void evaluateObjectCategory(const char *objectCategory, cv::Mat &normalMatrixNorm, std::array<float, 5> &anomalyDistributionNorm, PreProcessingPipeline &preProcessingPipeline) {

    std::string type[2] = {"Normal", "Anomaly"};
    for (int i = 0; i < 2; i++) {

        internal::averageNormalCells = 0; internal::averageAnomalyCells = 0;
        int normalCount = 0; int anomalyCount = 0;

        std::map<std::string, cv::Mat> images = readImagesFromDirectory("../data/chewinggum/Data/Images/" + type[i] + "/"); 

        std::cout << "\n";
        for (auto& [imageName, image] : images) {

            std::cout << "Evaluating image: " << imageName << " / " << images.size() << " :";

            preProcessingPipeline.apply(image);

            if ( internal::evaluateImage(image, normalMatrixNorm, anomalyDistributionNorm) )
                normalCount++;
            else 
                anomalyCount++;

        }

        std::cout << "you have called evaluationNormal" << "\n";
        std::cout << "Normal Predictions: (" << normalCount << "/" << images.size() << ") - avg normalCells(" << internal::averageNormalCells / images.size() << ")\n";
        std::cout << "Anomaly Predictions: (" << anomalyCount << "/" << images.size() << ") - avg anomalyCells(" << internal::averageAnomalyCells / images.size() << ")\n";
    }
}


namespace internal {

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

        int rowMargin = sampleImage.rows % internal::cellSize;
        int colMargin = sampleImage.cols % internal::cellSize;

        categoryNorm = cv::Mat::zeros((sampleImage.rows - rowMargin) / internal::cellSize, (sampleImage.cols - colMargin) / internal::cellSize, CV_32FC(5));
    }

    /*
     * evaluateImage
     * Determines wether an image is closer to given normal sample or a given anomally sample.
     * It is determined by a majority vote over each cell in the image.
     */
    bool evaluateImage(cv::Mat &image, cv::Mat &normalMatrix, std::array<float, 5> &anomalySample) {

        int rowMargin = image.rows % internal::cellSize;
        int colMargin = image.cols % internal::cellSize;

            int normalCellCount = 0;
            int anomalyCellCount = 0;

            int collIndex, rowIndex = 0; 
            for (int row = rowMargin / 2; row + internal::cellSize < image.rows - (rowMargin / 2); row += cellSize) {
                collIndex = 0;
                for (int col = colMargin / 2; col  + internal::cellSize < image.cols - (colMargin / 2); col += cellSize) {


                    // Get LBP distribution of cell
                    cv::Mat cell = image(cv::Range(row, row + internal::cellSize), cv::Range(col, col + cellSize));
                    std::array<float, 5> cellLBPHistogram = {};
                    lbpValueDistribution(cell, cellLBPHistogram);

                    // Get Normal & Anomaly Distance
                    float* normalSample = normalMatrix.ptr<float>(rowIndex,collIndex);

                    float normalDistance = 0, anomalyDistance = 0;
                    for (int i = 0; i < 5; i++) {
                        normalDistance += std::abs(cellLBPHistogram[i] - normalSample[i]);
                        anomalyDistance += std::abs(cellLBPHistogram[i] - anomalySample[i]);
                    }

                    // Classify Cell
                    if (normalDistance <= anomalyDistance) {
                        normalCellCount++;
                        internal::averageNormalCells++;

                    } else {
                        anomalyCellCount++; 
                        internal::averageAnomalyCells++;
                    }
                    cell.release();

                    collIndex++;
                }
                rowIndex++;
            }
            std::cout << " normalCellCount(" << normalCellCount << "), anomalyCellCount(" << anomalyCellCount << ")\n";


            if (normalCellCount > anomalyCellCount)
                return true;
            return false;
        }

}
