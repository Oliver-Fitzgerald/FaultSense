/*
 * pre-processing Manages the pre-processing flows for object and feature extraction 
 * steps
 */

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Fault Sense
#include "utils/beta/texture-beta.h"
#include "utils/generic-utils.h"
#include "utils/pre-processing-utils.h"
#include "objects/RGB.h"
#include "object-detection.h"
#include "feature-extraction.h"
#include "pre-processing.h"
#include "../common.h"
#include "../evaluation/evaluation.h"
#include "../training/train.h"
#include "../general/file-operations/training-data.h"
// Standard
#include <array>
#include <vector>

/*
 * markFaultLBP 
 */
bool markFaultLBP(std::vector<std::array<float, 5>>& normalSample, std::vector<std::array<float, 5>>& anomalySample, cv::Mat &image, std::string& imageCategory, const cv::Mat& imageMask, ObjectCoordinates& objectBounds) {

    int cellSize = 30;
    int rowMargin = image.rows % cellSize;
    int colMargin = image.cols % cellSize;

    if (std::size(normalSample) != std::size(anomalySample)) throw std::invalid_argument("normalSample and anomalySample size must be equal");


    for (int index = 0; index < normalSample.size(); index++) {

        int normalCells = 0;
        int anomalyCells = 0;
        float normalCellsCount = 0;
        float anomalyCellsCount = 0;
        float cells = 0;
        float totalNormalDistance = 0, totalAnomalyDistance = 0;
        bool result = false;
        bool lastResult;
        // Group cells to from histogram
        for (int row = (rowMargin * 2) + cellSize + objectBounds.xMin; row < image.rows - cellSize - (rowMargin * 2) - (cellSize / 2) - (image.rows - objectBounds.xMax); row += cellSize) {

            for (int col = (colMargin * 2) + cellSize + objectBounds.yMin; col < image.cols - cellSize - (colMargin * 2) - (cellSize / 2) - (image.cols - objectBounds.yMax); col += cellSize) {
                lastResult = result;

                // skip edges (in cases where there is to much noise at edge)
                if (imageCategory == "chewinggum")
                    if (row < cellSize * 2 || col < cellSize * 2 || row > ((image.rows - rowMargin / 2 - cellSize / 2) - cellSize) || col > ((image.cols - colMargin / 2 - cellSize / 2) - cellSize))
                        continue;


                cv::Mat maskCell = imageMask(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
                cv::Mat cell = image(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
                applyPreProcessing(cell, imageCategory, index);

                std::array<float, 5> cellLBPHistogram = {};
                lbpValueDistribution(cell, cellLBPHistogram);

                // Compare with normal and anomaly samples
                float normalDistance = 0; float anomalyDistance = 0;
                for (int i = 0; i < 5; i++) {
                    normalDistance += std::abs(cellLBPHistogram[i] - normalSample[index][i]);
                    anomalyDistance += std::abs(cellLBPHistogram[i] - anomalySample[index][i]);
                }

                if (isNormal(maskCell)) {
                    totalNormalDistance += normalDistance;
                    normalCells++;

                } else {
                    totalAnomalyDistance += anomalyDistance;
                    anomalyCells++;
                }

                // Evaluate cell
                bool result = classify(normalDistance, anomalyDistance, imageCategory, index);

                if (result)
                    normalCellsCount++;
                else
                    anomalyCellsCount++;

                // Mark anomaly
                if (!result) {
                    RGB colour = RGB{0,0,255};
                    markFault(image, col, col + cellSize, row , row + cellSize, nullptr, colour);
                }

                cells++;
            }
        }
        std::cout << "\033[34mINFO\033[0m: markfaultlbp complete index: " << index << "\n";
        std::cout << "\033[34mINFO\033[0m: totalNormalDistance: " << totalNormalDistance << "\n";
        std::cout << "\033[34mINFO\033[0m: totalanomalyDistance: " << totalAnomalyDistance << "\n";
        std::cout << "\033[34mINFO\033[0m: averageNormalDistance: " << totalNormalDistance / normalCells << "\n";
        std::cout << "\033[34mINFO\033[0m: averageAnomalyDistance: " << totalAnomalyDistance / anomalyCells << "\n";

        bool finalResult = true;
        if (imageCategory == "chewinggum") {
            if (((anomalyCellsCount / cells) * 100.0) > 0.5) finalResult = false;
            else finalResult = true;

        } else if (imageCategory == "cashew") {
            if (((anomalyCellsCount / cells) * 100.0) > 0.3) finalResult = false;
            else finalResult = true;
        }
        return finalResult;

    }
}
