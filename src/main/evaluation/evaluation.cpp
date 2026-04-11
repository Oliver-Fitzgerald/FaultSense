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
#include "evaluation-utils.h"
#include "../objects/PreProcessingPipeline.h"
#include "../objects/RGB.h"
#include "../objects/Features.h"
#include "../general/file-operations/generic-file-operations.h"
#include "../general/generic-utils.h"
#include "../../global-variables.h"


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

    EvaluationMetrics evaluationMetrics;

    std::string type[2] = {"Normal", "Anomaly"};
    for (int i = 0; i < 2; i++) {

        int normalCount = 0; int anomalyCount = 0;

        std::map<std::string, cv::Mat> images;
        readImagesFromDirectory(global::projectRoot + "data/chewinggum/Data/Images/" + type[i] + "/", images); 

        std::cout << "\n";
        for (auto& [imageName, image] : images) {

            std::cout << "Evaluating image: " << imageName << " / " << images.size() << " :";

            preProcessingPipeline.apply(image);

            if ( evaluate_utils::evaluateImage(image, normalMatrixNorm, anomalyDistributionNorm, evaluationMetrics) )
                normalCount++;
            else 
                anomalyCount++;

        }

        std::cout << "you have called evaluationNormal" << "\n";
        std::cout << "Normal Predictions: (" << normalCount << "/" << images.size() << ") - avg normalCells(" << evaluationMetrics.averageNormalCells / images.size() << ")\n";
        std::cout << "Anomaly Predictions: (" << anomalyCount << "/" << images.size() << ") - avg anomalyCells(" << evaluationMetrics.averageAnomalyCells / images.size() << ")\n";
        std::cout << " Avg Normal Distance: (" << evaluationMetrics.averageNormalDistance / images.size() << ")\n";
        std::cout << "Avg Anomaly Distance: (" << evaluationMetrics.averageAnomalyDistance / images.size() << ")\n";
    }
}

void markFaultLBP(const std::array<float, 5>& normalSample, const std::array<float, 5>& anomolySample, cv::Mat &image);

/*
 * markFaultLBP
 */
void markFaultLBP(const PreProcessingPipeline& preProcessingPipeline, const std::array<float, 5>& normalSample, const std::array<float, 5>& anomolySample, cv::Mat &image) {

    if (std::size(normalSample) != std::size(anomolySample)) throw std::invalid_argument("normalSample and anomolySample size must be equal");
    if (global::cellSize % 2 != 0) throw std::invalid_argument("cellSize must be a multiple of 2");

    // Group Cells to from histogram
    for (int row = global::cellSize / 2; row < image.rows - global::cellSize; row += global::cellSize) {
        for (int col = global::cellSize / 2; col < image.cols - global::cellSize; col += global::cellSize) {

            cv::Mat cell = image(cv::Range(row, row + global::cellSize), cv::Range(col, col + global::cellSize));
            std::array<float, 5> cellLBPHistogram = {};

            CannyThreshold threshold{57, 29};
            cv::Mat kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
            edgeDetection(cell, kernal, threshold);

            lbpValueDistribution(cell, cellLBPHistogram);

            // Compare with normal and anomoly samples
            float normalDistance = 0; float anomolyDistance = 0;
            for (int i = 0; i < std::size(normalSample); i++) {
                normalDistance += std::abs(cellLBPHistogram[i] - normalSample[i]);
                //std::cout << "- normalSample[" << i << "]: " << normalSample[i] << "\n";
                //std::cout << "- anomalySample[" << i << "]: " << anomolySample[i] << "\n";
                anomolyDistance += std::abs(cellLBPHistogram[i] - anomolySample[i]);
            }

            // Mark anomoly
            if (anomolyDistance < normalDistance) {
                //std::cout << "\nanomaly\n";
                RGB colour = RGB{0,0,255};
                markFault(image, col, col + global::cellSize, row , row + global::cellSize, nullptr, colour);
            } else;
                //std::cout << "\nnormal\n";
            //std::cout << "anomalyDistance : " << anomolyDistance << "\nnormalDistance: " << normalDistance << "\n";
        }
    }

    /* Testing
    cv::imshow("Image", image);
    while (cv::pollKey() != 113);
    */
}

/*
 * markFaultLBP
 */
void markFaultLBP(FeatureFilter& cellFeature, PreProcessingPipeline& preProcessingPipeline, cv::Mat& normalSample, const std::array<float, 5>& anomolySample, cv::Mat &image) {

    //if (std::size(normalSample) != std::size(anomolySample)) throw std::invalid_argument("normalSample and anomolySample size must be equal");
    if (global::cellSize % 2 != 0) throw std::invalid_argument("cellSize must be a multiple of 2");

    cv::Mat returnImage = image.clone();
    ObjectCoordinates objectBounds;

    // Apply objectDetection if relevant
    if (preProcessingPipeline.objectDetectionConfiguration.has_value()) {

        std::optional<PreProcessing> preProcessingConfiguration = preProcessingPipeline.preProcessingConfiguration;
        std::optional<PreProcessing> objectDetectionConfiguration = preProcessingPipeline.objectDetectionConfiguration;

        objectDetectionConfiguration->apply(image);
        objectBounds = getObject(image);

        image = returnImage.clone();
        preProcessingConfiguration->apply(image);
    }

    // Itterate over image cells
    int collIndex, rowIndex = 0; 
    for (int row = (global::cellSize / 2) + objectBounds.xMin + global::cellSize; row < (image.rows - global::cellSize) - (image.rows - objectBounds.xMax) - global::cellSize; row += global::cellSize) {
        collIndex = 0;
        for (int col = (global::cellSize / 2) + objectBounds.yMin + global::cellSize; col < (image.cols - global::cellSize) - (image.cols - objectBounds.yMax) - global::cellSize; col += global::cellSize) {

            cv::Mat cell = image(cv::Range(row, row + global::cellSize), cv::Range(col, col + global::cellSize));
            std::array<float, 5> cellLBPHistogram = {};

            // Feature Extraction
            cellFeature.extractFeature(cell);

            int whitePixelCount = evaluate_utils::countWhitePixels(cell);
            lbpValueDistribution(cell, cellLBPHistogram);

            //Cell evaluation
            float* normal = normalSample.ptr<float>(rowIndex,collIndex);
            float normalDistance = 0; float anomolyDistance = 0;
            for (int i = 0; i < 5; i++) {
                normalDistance += std::abs(cellLBPHistogram[i] - normal[i]);
                anomolyDistance += std::abs(cellLBPHistogram[i] - anomolySample[i]);
                //std::cout << "- normalSample[" << i << "]: " << normal[i] << "\n";
                //std::cout << "- anomalySample[" << i << "]: " << anomolySample[i] << "\n";
            }

            // Mark anomoly
            //if (anomolyDistance < normalDistance) {
            if (whitePixelCount > 100) {
                //std::cout << "\nanomaly\n";
                RGB colour = RGB{0,0,255};
                markFault(returnImage, col, col + global::cellSize, row , row + global::cellSize, nullptr, colour);
            } else
                //std::cout << "\nnormal\n";
            //imageViewer(cell);
            std::cout << "anomalyDistance : " << anomolyDistance << "\nnormalDistance: " << normalDistance << "\n";
            collIndex++;
        }
        rowIndex++;
    }

    image = returnImage.clone();

    /* Testing
    cv::imshow("Image", image);
    while (cv::pollKey() != 113);
    */
}

/*
 * checkIfCellIsNormal
 * Checks if a cell contains an anomaly i.e a white cell (binary anomaly detection)
 *
 * @param cell (cv::Mat) the cell to be checked for an anomaly
 * @return normal (bool) A boolean indicating wether or not the cell was normal (true)
 *                       or contained an anomaly (false)
 */
bool checkIfCellIsNormal(cv::Mat cell) {

    for (int row = 0; row < cell.rows; row++) { 
        for (int col = 0; col < cell.cols; col++) {
            
            int pixel = cell.at<uchar>(row, col);
            if (pixel == 255) // white
                return false;
        }
    }

    return true;
}
