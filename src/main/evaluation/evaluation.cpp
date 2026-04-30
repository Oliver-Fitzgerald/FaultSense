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
#include "../feature/utils/generic-utils.h"
#include "../feature/feature-extraction.h"
#include "../feature/pre-processing.h"
#include "../feature/object-detection.h"
#include "../objects/PixelCoordinates.h"
#include "../objects/ConfusionMatrix.h"
#include "../common.h"

void evaluateNormal(const char objectCategory, cv::Mat &normalNormSample, std::array<float, 5> &anomalyDistribution);
void initMatrix(const std::map<std::string, cv::Mat>::iterator &itterator, int cellSize, cv::Mat &categoryNorm);
void initMatrix(cv::Mat &sampleImage, int cellSize, cv::Mat &categoryNorm);
bool compareToNorm(cv::Mat norm, std::array<float, 5> &anomalySample, cv::Mat values, int cellSize);


int averageNormalCells = 0;
int averageAnomalyCells = 0;

void evaluateNormal(const char *objectCategory, cv::Mat &normalNormSample, std::array<float, 5> &anomalyDistribution) {


    int cellSize = 60;


    std::string type[2] = {"Normal", "Anomaly"};
    for (int i = 0; i < 2; i++) {

        averageNormalCells = 0; averageAnomalyCells = 0;
        int normalCount = 0; int anomalyCount = 0;
        std::map<std::string, cv::Mat> images = readImagesFromDirectory("../data/chewinggum/Data/Images/" + type[i] + "/"); 
        std::cout << "\n";
        for (auto& [imageName, image] : images) {

            std::cout << "Evaluating image: " << imageName << " / " << images.size() << " :";

            // Object Detection
            cv::Mat object;
            std::string name = imageName;
            ObjectCoordinates objectBounds;
            objectDetection(image, object, name, objectBounds);
            
            // Compute LBP values for each pixel
            cv::Mat LBPValues;
            lbpValues(object, LBPValues);

            if ( compareToNorm(normalNormSample, anomalyDistribution, LBPValues, cellSize) )
                normalCount++;
            else 
                anomalyCount++;

        }

        std::cout << "you have called evaluationNormal" << "\n";
        std::cout << "Normal Predictions: (" << normalCount << "/" << images.size() << ") - avg normalCells(" << averageNormalCells / images.size() << ")\n";
        std::cout << "Anomaly Predictions: (" << anomalyCount << "/" << images.size() << ") - avg anomalyCells(" << averageAnomalyCells / images.size() << ")\n";
    }
}


void initMatrix(const std::map<std::string, cv::Mat>::iterator &itterator, int cellSize, cv::Mat &categoryNorm) {

    cv::Mat sampleImage = itterator->second; cv::Mat sampleValues;
    lbpValues(sampleImage, sampleValues); // To remove 1 pixel margin

    int rowMargin = sampleValues.rows % cellSize;
    int colMargin = sampleValues.cols % cellSize;

    categoryNorm = cv::Mat::zeros((sampleValues.rows - rowMargin) / cellSize, (sampleValues.cols - colMargin) / cellSize, CV_32FC(5));
}

void initMatrix(cv::Mat &sampleImage, int cellSize, cv::Mat &categoryNorm) {

    cv::Mat sampleValues;
    lbpValues(sampleImage, sampleValues); // To remove 1 pixel margin

    int rowMargin = sampleValues.rows % cellSize;
    int colMargin = sampleValues.cols % cellSize;

    categoryNorm = cv::Mat::zeros((sampleValues.rows - rowMargin) / cellSize, (sampleValues.cols - colMargin) / cellSize, CV_32FC(5));
}

/*
 * compareToNorm
 */
bool compareToNorm(cv::Mat norm, std::array<float, 5> &anomalySample, cv::Mat values, int cellSize) {

    int rowMargin = values.rows % cellSize;
    int colMargin = values.cols % cellSize;

    int normalCount = 0;
    int anomalyCount = 0;

    int collIndex, rowIndex = 0; 
    for (int row = rowMargin / 2; row + cellSize < values.rows - (rowMargin / 2); row += cellSize) {
        collIndex = 0;
        for (int col = colMargin / 2; col  + cellSize < values.cols - (colMargin / 2); col += cellSize) {

            /* DEBUGING: std::cout << "\nvalus.rows: " << values.rows << "\n"; std::cout << "valus.cols: " << values.cols << "\n"; std::cout << "rowRange(from, to): (" << row << ", " << row + cellSize << ")\n"; std::cout << "colRange(from, to): (" << col << ", " << col + cellSize << ")\n"; */

            // Get LBP distribution of cell
            cv::Mat cell = values(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
            std::array<float, 5> cellLBPHistogram = {};
            lbpValueDistribution(cell, cellLBPHistogram);

            float* normalSample = norm.ptr<float>(rowIndex,collIndex);

            float normalDistance = 0, anomalyDistance = 0;
            for (int i = 0; i < 5; i++) {
                normalDistance += std::abs(cellLBPHistogram[i] - normalSample[i]);
                anomalyDistance += std::abs(cellLBPHistogram[i] - anomalySample[i]);
            }

            // Update the cell norm with the samples value
            if (normalDistance <= anomalyDistance) {
                normalCount++;
                averageNormalCells++;

            } else {
                anomalyCount++; 
                averageAnomalyCells++;
            }
            cell.release();

            collIndex++;
        }
        rowIndex++;
    }
    std::cout << " normalCount(" << normalCount << "), anomalyCount(" << anomalyCount << ")\n";

    /*
    float* normal = norm.ptr<float>(3,3);
    std::array<float, 5> normalSample;

    for (int i = 0; i < 5; i++)
        normalSample[i] = normal[i];
    markFaultLBP(normalSample, anomalySample, values);
    */

    if (normalCount > anomalyCount)
        return true;
    return false;
}




bool evaluate(ConfusionMatrix& confusionMatrix, std::string& category, std::vector<std::array<float,5>>& normalSample, std::vector<std::array<float,5>>& anomalySample, cv::Mat& image, cv::Mat& imageMask) {

    // Object Detection
    cv::Mat inputTemp = image.clone();
    cv::Mat outputTemp;
    ObjectCoordinates objectBounds;

    objectDetection(inputTemp, outputTemp, category, objectBounds);
    crop(inputTemp, objectBounds.xMin, objectBounds.xMax, objectBounds.yMin, objectBounds.yMax, image);

    if (!imageMask.empty()) {
        cv::Mat maskCopy = imageMask.clone();
        crop(maskCopy, objectBounds.xMin, objectBounds.xMax, objectBounds.yMin, objectBounds.yMax, maskCopy);
        imageMask = maskCopy;
    }


    // Evaluation
    int cellSize = 30;
    int rowMargin = image.rows % cellSize;
    int colMargin = image.cols % cellSize;

    if (std::size(normalSample) != std::size(anomalySample)) throw std::invalid_argument("normalSample and anomalySample size must be equal");

    int normalCells = 0, anomalyCells = 0;
    double cells = 0;
    for (int index = 0; index < normalSample.size(); index++) {

        for (int row = rowMargin / 2; row < image.rows - rowMargin / 2 - cellSize / 2; row += cellSize) {
            for (int col = colMargin / 2; col < image.cols - colMargin / 2 - cellSize / 2; col += cellSize) {
                cells++;

                // skip edges (in cases where there is to much noise at edge)
                if (category == "chewinggum")
                    if (row < cellSize * 2 || col < cellSize * 2 || row > ((image.rows - rowMargin / 2 - cellSize / 2) - cellSize) || col > ((image.cols - colMargin / 2 - cellSize / 2) - cellSize))
                        continue;

                cv::Mat cell = image(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
                applyPreProcessing(cell, category, index);

                std::array<float, 5> cellLBPHistogram = {};
                lbpValueDistribution(cell, cellLBPHistogram);

                // Compare with normal and anomaly samples
                float normalDistance = 0; float anomalyDistance = 0;
                for (int i = 0; i < 5; i++) {
                    normalDistance += std::abs(cellLBPHistogram[i] - normalSample[index][i]);
                    anomalyDistance += std::abs(cellLBPHistogram[i] - anomalySample[index][i]);
                }

                // Evaluate Classification
                bool result = classify(normalDistance, anomalyDistance, category, index);

                if (!imageMask.empty()) { // Anomaly

                    CV_Assert(imageMask.type() == CV_8UC1);
                    CV_Assert(imageMask.size() == image.size());

                    cv::Mat maskCell = imageMask(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
                    confusionMatrix.update(result, isNormal(maskCell));

                } else // Normal
                    confusionMatrix.update(result, true);


                if (result) normalCells++;
                else        anomalyCells++;
            }
        }

    }


    // if (imageMask.empty()) std::cout << "Normal Sample: ";
    // else std::cout << "Anomaly Sample: ";
    //
    // std::cout << "anomalyCells(" << ((anomalyCells / cells) * 100) << "), normalCells(" << ((normalCells / cells) * 100) << ")\n";

    bool result;
    if (category == "chewinggum") {
        if (((anomalyCells / cells) * 100) > 0.5) result = false;
        else result = true;
    } else if (category == "cashew") {
        if (((anomalyCells / cells) * 100) > 0.3) result = false;
        else result = true;
    }

    // std::cout << "normalCells(" << normalCells << "), anomalyCells(" << anomalyCells << ") " << result << "\n";

    return result;
}


bool evaluate(ConfusionMatrix& confusionMatrix, std::string& category, std::vector<std::array<float,5>>& normalSample, std::vector<std::array<float,5>>& anomalySample, cv::Mat& image) {
    cv::Mat dummy = cv::Mat();
    return evaluate(confusionMatrix, category, normalSample, anomalySample, image, dummy);
}
