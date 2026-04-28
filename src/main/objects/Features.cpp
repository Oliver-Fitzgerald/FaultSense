// Standard
#include <typeinfo>
#include <string>
// OpenCV2
#include <opencv2/core.hpp>
// Fault Sense
#include "../evaluation/evaluation.h"
#include "../frontend/cli/image-viewer/image-viewer.h"
#include "../general/file-operations/image-file-operations.h"


class FeatureFilter {

public:

    const int cellSize = 60;
    cv::Mat featureMatrix;
    bool featureInitalized = false;
    int rowMargin;
    int colMargin;


    virtual void extractFeature(cv::Mat& cell, bool singleCell, const bool anomaly, std::string imageName) = 0;
    virtual void updateFeature(cv::Mat& cell, bool singleCell, const bool anomaly = false, std::string imageName = "None") = 0;

    virtual std::string getName() = 0;

    virtual ~FeatureFilter() = default;

    /*
     * initFeatureMatrix
     * Initalizes a features value matrix
     * @param sampleImage   -
     */
    void initFeatureMatrix(const cv::Mat& sampleImage) {

        int rowMargin = sampleImage.rows % cellSize;
        int colMargin = sampleImage.cols % cellSize;

        featureMatrix = cv::Mat::zeros((sampleImage.rows - rowMargin) / cellSize, (sampleImage.cols - colMargin) / cellSize, CV_64F);
    }

    /*
     * reset
     * Sets a flag to indicate that the implmentation of updateFeature should re-initzlie feature
     * extraction values on it's next run i.e new data has been received re-train features
     */
    void reset() {
        this->featureInitalized = false;
    }

};


/*
 * BinaryCountFeature
 * Feature describing the ratio of white pixels present in a binary image Stores the ratio of white to non-white pixels in each cell/image as the percentage of white pixels
 * Note: Image/Cell must only contain 1 channel and be thersholded to 0 | 255
 */
class BinaryCountFeature : public FeatureFilter {
private:

    /*
     * extractFeature
     * Extracts the ratio of white pixels in the passed (image/image cells) as a percentage of 100
     * @param image - The image that the white pixels will be counted in
     */
    void extractFeature(cv::Mat& image, bool singleCell, const bool anomaly, std::string imageName = "None") override {

        cv::Mat imageMask;
        if (anomaly) { // read image mask
            if (imageName == "None") throw std::invalid_argument("optional parameter imageName must be passed if anomaly true");

            const std::string from = "Images";
            const std::string to = "Masks";

            std::size_t pos = imageName.find(from);
            if (pos != std::string::npos)
                imageName.replace(pos, from.length(), to);
            else
                throw std::invalid_argument("imageName must be the full path: " + imageName);

            pos = imageName.rfind(".JPG");
            if (pos != std::string::npos)
                imageName.replace(pos, 4, ".png");
            else
                throw std::invalid_argument("imageName must be the full path: " + imageName);

            readImage(imageName, imageMask);
        }

        int colIndex, rowIndex = 0; 
        for (int row = (rowMargin / 2); (row + cellSize) < image.rows - (rowMargin / 2); row += cellSize) {
            colIndex = 0;
            for (int col = colMargin / 2; (col  + cellSize) < image.cols - (colMargin / 2); col += cellSize) {

                cv::Mat cell = image(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));
                cv::Mat cellMask;
                if (anomaly) cellMask = imageMask(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));

                if (singleCell) {
                    if (anomaly && !checkIfCellIsNormal(cellMask))
                        featureMatrix.at<double>(0, 0) += updatePixelValue(cell);
                    std::cout << "value(0, 0):" << featureMatrix.at<double>(0, 0) << "\n";
                } else {

                    /* DEBUG INFO (UNCOMMENT) 
                    std::cout << "rowIndex: " << rowIndex << "\n";
                    std::cout << "feature.rows: " << featureMatrix.rows << "\n";
                    std::cout << "colIndex: " << colIndex << "\n";
                    std::cout << "feature.cols: " << featureMatrix.cols << "\n\n";
                    */
                    std::cout << "value before pixelUpdate(" << rowIndex << ", " << colIndex << "):" << featureMatrix.at<double>(rowIndex, colIndex) << "\n";
                    featureMatrix.at<double>(rowIndex, colIndex) += updatePixelValue(cell);
                    std::cout << "value after pixelUpdate(" << rowIndex << ", " << colIndex << "):" << featureMatrix.at<double>(rowIndex, colIndex) << "\n";
                }

                cell.release();
                colIndex++;
            }
            rowIndex++;
        }
    }

public:

    double pixelWeigth;
    int pixelCount;


    /*
     * updateFeature
     * Updates the current feature to incorporate the pixel values in the passed image
     * @param image - The image providing pixel values for feature update
     */
    void updateFeature(cv::Mat& image, bool singleCell, const bool anomaly, std::string imageName = "None") override {

        if (anomaly && singleCell != true) {
            throw std::invalid_argument("If you are updating an anomaly feature singleCell should be true");
            //singleCell = true;
        }

        if (!featureInitalized) {

            this->rowMargin = image.rows % cellSize;
            this->colMargin = image.cols % cellSize;

            this->pixelCount = cellSize * cellSize;

            if (singleCell) {
                this->featureMatrix = cv::Mat(1, 1, CV_64F);
            } else {
                initFeatureMatrix(image);
            }
            std::cout << "pixelCount: " << pixelCount << "\n";
            this->pixelWeigth = 100.0 / (this->pixelCount + 1);
            std::cout << "pixelWeigth: " << pixelWeigth << "\n";

            this->extractFeature(image, singleCell, anomaly, imageName);
            this->featureInitalized = true;

        } else {

            // Update current feature values
            pixelCount *= pixelCount;

            double oldPixelWeigth = pixelWeigth;
            pixelWeigth = 100.0 / (pixelCount + 1);

            double scale = oldPixelWeigth / pixelWeigth;

            for (int row = 0; row < featureMatrix.rows; row++)
                for (int col = 0; col < featureMatrix.cols; col++)
                    featureMatrix.at<double>(row, col) *= scale;
            std::cout << "pixelWeigth (after scale): " << pixelWeigth << "\n";

            // Append new pixel values to feature values
            this->extractFeature(image, singleCell, anomaly, imageName);
        }
    }

    std::string getName() {
        return std::string("BinaryCountFeature");
    }


    /*
     * updatePixelValue
     * Updates a feature value based on the current pixelValue
     * @param featureValue - Cell providing pixel values for feature update
     * @param featureValue - The value of the current feature
     */
    double updatePixelValue(cv::Mat& cell) {


        std::cout << "pixelWeigth: " << pixelWeigth << "\n";
        std::cout << "cel.row: " << cell.rows << "\n";
        std::cout << "cel.row: " << cell.cols << "\n";
        double featureValue = 0;
        int count = 0;
        ////////////////////////////////////////////////
        while (cv::pollKey() != 113) cv::imshow("Imag", cell);
        ////////////////////////////////////////////////
        for (int row = 0; row < cell.rows; row++) {
            for (int col = 0; col < cell.cols; col++) {

                count++;
                //std::cout << "pixel value: " << (int)cell.at<uchar>(row, col) << "\n";
                if (cell.at<uchar>(row, col) > 0)
                     featureValue += pixelWeigth;

            }
        }
        std::cout << "returning feature value: " << featureValue << "\n";
        std::cout << "count: " << count / pixelWeigth << "\n\n";
        return featureValue;
    }

};

/*
 * BinaryDistributionFeature
 * Describes the distribution of binary pixel values in an image
 * Note: Image must be of only 1 channel with values between 0 and 255 inclusive 
 */
class BinaryDistributionFeature : public FeatureFilter {

private:
    void extractFeature(cv::Mat& image, bool singleCell, const bool anomaly, std::string imageName = "") {

        int colIndex, rowIndex = 0; 
        for (int row = (rowMargin / 2); (row + cellSize) < image.rows - (rowMargin / 2); row += cellSize) {
            colIndex = 0;
            for (int col = colMargin / 2; (col  + cellSize) < image.cols - (colMargin / 2); col += cellSize) {

                cv::Mat cell = image(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));

                if (singleCell)
                    updatePixelValue(cell);
                else {
                    updatePixelValue(cell);
                }

                cell.release();
                colIndex++;
            }
            rowIndex++;
        }
    }

public:

    void updateFeature(cv::Mat& image, bool singleCell, const bool anomaly, std::string imageName = "None") {

        if (anomaly) singleCell = true;

        if (!featureInitalized) {

            this->rowMargin = image.rows % cellSize;
            this->colMargin = image.cols % cellSize;

            if (singleCell) {
                this->featureMatrix = cv::Mat(1, 1, CV_64F);

            } else {
                initFeatureMatrix(image);
            }

            this->extractFeature(image, singleCell, anomaly);
            this->featureInitalized = true;

        } else {

            // Update current feature values
            double scale = 0;

            for (int row = 0; row < featureMatrix.rows; row++)
                for (int col = 0; col < featureMatrix.cols; col++)
                    featureMatrix.at<double>(row, col) *= scale;

            // Append new pixel values to feature values
            this->extractFeature(image, singleCell, anomaly);
        }
    }

    std::string getName() {
        return std::string("BinaryDistributionFeature");
    }

    /*
     * updatePixelValue
     * Updates the given feature value based on the cells image data
     * In this case with the distribution of pixel values
     * @param cell          -
     * @param featureValue  -
     */
    double updatePixelValue(cv::Mat& cell) {

        return 0.0;
    }
};

