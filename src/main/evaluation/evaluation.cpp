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

void evaluateNormal(const char objectCategory, cv::Mat &normalNormSample, std::array<float, 5> &anomalyDistribution);
void initMatrix(const std::map<std::string, cv::Mat>::iterator &itterator, int cellSize, cv::Mat &categoryNorm);
void initMatrix(cv::Mat &sampleImage, int cellSize, cv::Mat &categoryNorm);
bool compareToNorm(cv::Mat norm, std::array<float, 5> &anomalySample, cv::Mat values, int cellSize);


int averageNormalCells = 0;
int averageAnomalyCells = 0;

/*
 * evaluateNormal
 */
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
            objectDetection(image, object);
            
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

            //std::cout << "(row, col) => (" << row << ", " << col << "\n";

            /* DEBUG INFO
            std::cout << "\nvalus.rows: " << values.rows << "\n";
            std::cout << "valus.cols: " << values.cols << "\n";
            std::cout << "rowRange(from, to): (" << row << ", " << row + cellSize << ")\n";
            std::cout << "colRange(from, to): (" << col << ", " << col + cellSize << ")\n";
            */
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

