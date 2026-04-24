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
#include "../pre-processing/object-detection.h"
#include "../classification/ClassificationModel.h"
#include "../general/file-operations/image-file-operations.h"
#include "../general/generic-utils.h"
#include "../../global-variables.h"


/*
 * evaluateObjectCategory
 * Given an object category it iterates over all normal and anomoly sample from the given category and prints 
 * the evaluation of each image as well as the total and average evaluations for normal and anomaly samples
 * @param objectCategory
 */
void evaluateObjectCategory(const std::string& objectCategory, FeaturesCollection& features, std::map<std::string, cv::Mat>& normalFeatures, std::map<std::string, cv::Mat>& anomalyFeatures) {


    std::cout << "######################################\n";
    std::cout << "# Evaluation Results\n";
    std::cout << "######################################\n";

    EvaluationMetrics evaluationMetrics;

    std::string type[2] = {"Normal", "Anomaly"};
    for (int i = 0; i < 2; i++) {

        int normalCount = 0; int anomalyCount = 0;

        std::map<std::string, cv::Mat> images;
        readImagesFromDirectory(global::projectRoot + "data/" + objectCategory + "/Data/Images/" + type[i] + "/", images); 

        std::cout << "\n";
        for (auto& [imageName, image] : images) {

            std::cout << "Evaluating image: " << imageName << " / " << images.size() << " :";

            if ( evaluate_utils::evaluateImage(image, features, normalFeatures, anomalyFeatures, evaluationMetrics) )
                normalCount++;
            else 
                anomalyCount++;

        }

        // Print Evaluation Report
        std::cout << "you have called evaluationNormal" << "\n";
        std::cout << "Normal Predictions: (" << normalCount << "/" << images.size() << ") - avg normalCells(" << evaluationMetrics.averageNormalCells / images.size() << ")\n";
        std::cout << "Anomaly Predictions: (" << anomalyCount << "/" << images.size() << ") - avg anomalyCells(" << evaluationMetrics.averageAnomalyCells / images.size() << ")\n";
        std::cout << " Avg Normal Distance: (" << evaluationMetrics.averageNormalDistance / images.size() << ")\n";
        std::cout << "Avg Anomaly Distance: (" << evaluationMetrics.averageAnomalyDistance / images.size() << ")\n";
    }
}

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
 * markFaults
 * Given a set of normal and anomaly features this function extracts those same features from the given image
 * and classifies sectors of the image as anomalies or normal based of those passed features
 * @param normalFeatures    - 
 * @param anomalyFeatures   -
 * @param image             -
 */
void markFaults(std::map<std::string, cv::Mat>& normalFeatures, std::map<std::string, cv::Mat>& anomalyFeatures, cv::Mat &image, FeaturesCollection& featureCollection, std::string imageName) {

    cv::Mat imageTemp = image.clone();
    std::map<std::string, cv::Mat> features;
    featureCollection.train(image, false, imageName).extract(features);

    image = imageTemp.clone();

    //if (std::size(normalSaple) != std::size(anomolySample)) throw std::invalid_argument("normalSample and anomolySample size must be equal");
    if (global::cellSize % 2 != 0) throw std::invalid_argument("cellSize must be a multiple of 2");

    int rowMargin = image.rows % global::cellSize;
    int colMargin = image.cols % global::cellSize;
    std::cout << "initalizing classification model ... \n";
    std::unique_ptr<ClassificationModel> model = std::make_unique<SVM>(normalFeatures, anomalyFeatures);
    std::cout << "classification model initalized\n";


    BinaryCountFeature temp = BinaryCountFeature();
    temp.initFeatureMatrix(image);
    cv::Mat featureMatrix = temp.featureMatrix;


    ObjectCoordinates objectBounds;
    auto& pipeline = featureCollection.features.begin()->second;
    cv::Mat temp3 = image.clone();
    if (pipeline->objectDetectionConfiguration->applyObjectDetection) {

        cv::Mat mat = image.clone();
        cv::Mat temp1 = image.clone();
        cv::Mat temp2 = image.clone();
        pipeline->objectDetectionConfiguration->apply(temp1, &objectBounds);
        objectDetection(temp2, temp3, objectBounds);

    } else {
        objectBounds = ObjectCoordinates{.xMin=0, .xMax=image.rows, .yMin=0, .yMax=image.cols};
    }

    std::vector<std::vector<svm_node>> featureValues = model->toSvmNodes(features);
    auto featureItterator = featureValues.begin();



    auto& mat = featureCollection.features.begin()->first->featureMatrix;
    std::cout << " rows=" << mat.rows 
              << " cols=" << mat.cols 
              << " channels=" << mat.channels()
              << " total=" << mat.total() << "\n";

    std::cout << "grid cells: " << (image.rows / global::cellSize) 
              << " x " << (image.cols / global::cellSize) << "\n";
    std::cout << "featureMat: " << mat.rows << " x " << mat.cols << "\n";

    int expectedCells = 0;
    for (int row = 0; row < mat.rows; ++row) {
        for (int col = 0; col < mat.cols; ++col) {
            int cellIdx = row * mat.cols + col;
            bool result = model->classify(featureValues[cellIdx]);
            if (!result) {
                int imgRow = row * global::cellSize;
                int imgCol = col * global::cellSize;
                RGB colour = RGB{0, 0, 255};
                markFault(temp3, imgCol, imgCol + global::cellSize, 
                                 imgRow, imgRow + global::cellSize, nullptr, colour);
            }
        }
    }

    image = temp3.clone();

    std::cout << "featureValues.size()=" << featureValues.size() 
          << " expectedCells=" << expectedCells << "\n";

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


    // while (cv::pollKey() != 113) cv::imshow("image", cell);
    // std::cout << "======================================\n";
    // std::cout << "channels: " << cell.channels() << "\n";
    // std::cout << "type: " << cell.type() << "\n";
    // std::cout << "depth: " << cell.depth() << "\n";
    //
    // uchar b = pixel[0];
    // uchar g = pixel[1];
    // uchar r = pixel[2];
    // int ib = pixel[0];
    // int ig = pixel[1];
    // int ir = pixel[2];
    //
    // std::cout << "b: " << (int)b << "\n";
    // std::cout << "g: " << (int)g << "\n";
    // std::cout << "r: " << (int)r << "\n";
    // std::cout << "b: " << ib << "\n";
    // std::cout << "g: " << ig << "\n";
    // std::cout << "r: " << ir << "\n";
    // std::cout << "empty? " << cell.empty() << "\n";
    // std::cout << "size: " << cell.cols << " x " << cell.rows << "\n";
    // std::cout << "======================================\n";
    

    for (int row = 0; row < cell.rows; row++) { 
        for (int col = 0; col < cell.cols; col++) {
            
            int pixel = cell.at<uchar>(row, col);

            // std::cout << "value(" << row << ", " << col << "): " << pixel << "\n";
            cv::Vec3b testPixel = cell.at<cv::Vec3b>(row, col);
            // std::cout << "value(" << row << ", " << col << "): " << (int)testPixel[0];
            // std::cout << ", " << (int)testPixel[1];
            // std::cout << ", " << (int)testPixel[2] << "\n";;

            if (pixel == 255) // white
                return false;
        }
    }
    // std::cout << "======================================\n";

    return true;
}
