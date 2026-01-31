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

void evaluateNormal(const char objectCategory, cv::Mat &normalNormSample, std::array<float, 5> &anomalyDistribution);
void initMatrix(const std::map<std::string, cv::Mat>::iterator &itterator, int cellSize, cv::Mat &categoryNorm);
bool compareToNorm(cv::Mat norm, std::array<float, 5> &anomalySample, cv::Mat values, int cellSize);

void evaluateNormal(const char *objectCategory, cv::Mat &normalNormSample, std::array<float, 5> &anomalyDistribution) {


    std::map<std::string, cv::Mat> images = readImagesFromDirectory("../../../data/chewinggum/Data/Images/Normal/"); 
    int cellSize = 60;

    int normalCount = 0; int anomalyCount = 0;
    std::cout << "\n";
    for (const auto& [imageName, image] : images) {
        std::cout << "Evaluating image: " << imageName << " / " << images.size() << " :";

        cv::Mat sample;
        initMatrix(images.begin(), cellSize, sample);

        // Compute LBP values for each pixel
        cv::Mat LBPValues;
        lbpValues(image, LBPValues);

        if ( compareToNorm(normalNormSample, anomalyDistribution, LBPValues, cellSize) )
            normalCount++;
        else 
            anomalyCount++;
        // left off here
    }

    std::cout << "you have called evaluationNormal" << "\n";
    std::cout << "Normal Predictions: (" << normalCount << "/" << images.size() << ")\n";
    std::cout << "Anomaly Predictions: (" << anomalyCount << "/" << images.size() << ")\n";
}


void initMatrix(const std::map<std::string, cv::Mat>::iterator &itterator, int cellSize, cv::Mat &categoryNorm) {

    cv::Mat sampleImage = itterator->second; cv::Mat sampleValues;
    lbpValues(sampleImage, sampleValues); // To remove 1 pixel margin

    int rowMargin = sampleValues.rows % cellSize;
    int colMargin = sampleValues.cols % cellSize;

    categoryNorm = cv::Mat::zeros((sampleValues.rows - rowMargin) / cellSize, (sampleValues.cols - colMargin) / cellSize, CV_32FC(5));
}

bool compareToNorm(cv::Mat norm, std::array<float, 5> &anomalySample, cv::Mat values, int cellSize) {

    int rowMargin = values.rows % cellSize;
    int colMargin = values.cols % cellSize;

    int normalCount = 0;
    int anomalyCount = 0;

    int collIndex, rowIndex = 0; 
    for (int row = rowMargin / 2; row < values.rows - (rowMargin / 2); row += cellSize) {
        collIndex = 0;
        for (int col = colMargin / 2; col < values.cols - (colMargin / 2); col += cellSize) {

            // Get LBP distribution of cell
            cv::Mat cell = values(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
            std::array<float, 5> LBPHistogram = {};
            lbpValueDistribution(cell, LBPHistogram);

            float* normalSample = norm.ptr<float>(rowIndex,collIndex);

            float normalDistance = 0, anomalyDistance = 0;
            for (int i = 0; i < 5; i++) {
                normalDistance += std::abs(LBPHistogram[i] - normalSample[i]);
                anomalyDistance += std::abs(LBPHistogram[i] - anomalySample[i]);
            }

            // Update the cell norm with the samples value
            if (normalDistance <= anomalyDistance)
                normalCount++;
            else
                anomalyCount++; 

            collIndex++;
        }
        rowIndex++;
    }

    std::cout << " normalCount(" << normalCount << "), anomalyCount(" << anomalyCount << ")\n";


    if (normalCount > anomalyCount)
        return true;
    return false;
}
