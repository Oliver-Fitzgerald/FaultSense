#ifndef train_internal_H
#define train_internal_H

// Fault Sense
#include "../objects/PreProcessingPipeline.h"

namespace internal {

    const std::string dataRoot = "../data/";
    const std::string objectCategories[12] = {
        "chewinggum/",
        "candle/",
        "capsules/",
        "cashew/",
        "fryum/",
        "macaroni1/",
        "macaroni2/",
        "pcb1/",
        "pcb2/",
        "pcb3/",
        "pcb4/",
        "pipe_fryum/"
    };
    const std::string anomalyPath = "Data/Images/Anomaly";
    const std::string normalPath = "Data/Images/Normal";
    const int cellSize = 60; // shoulde be divisible by 2

    void initNormMatrix(const std::map<std::string, cv::Mat>::iterator &itterator, int cellSize, cv::Mat &categoryNorm);
    void initNormMatrix(const cv::Mat &sampleImage, int cellSize, cv::Mat &categoryNorm);
    void updateCategoryNorm(cv::Mat norm, cv::Mat values, int cellSize, int numberOfSamples);
    void generateNormalCellNorm(std::array<float, 5> &cellNorm, std::vector<cv::Mat> &images, const PreProcessingPipeline &preProcessingConfiguration);
    void generateAnomalyCellNorm(std::array<float, 5> &cellNorm, std::map<std::string, cv::Mat> &images, const PreProcessingPipeline &preProcessingConfiguration, const std::string &categoryName);
}

#endif
