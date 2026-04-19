/*
 * Features
 * Contains feature classes used to store a arbitrary feature data object as well
 * as transformer feature functions for extracting and compare features from images (cv::Mat)
 *
 * Contains the following class
 * FeatureFilter - Interface for all Features
 * BinaryCountFeature - TODO
 * BinaryDistributionFeature - TODO
 */
#ifndef Features_H
#define Features_H

// Standard
#include <typeinfo>
// OpenCV2
#include <opencv2/core.hpp>

class FeatureFilter {

public:

    const int cellSize = 60;
    int rowMargin;
    int colMargin;

    virtual void extractFeature(cv::Mat& cell) = 0;
    virtual void updateFeature(cv::Mat& cell) = 0;
    virtual double compare(FeatureFilter* feature) = 0;

    virtual ~FeatureFilter() = default;
};


/*
 * BinaryCountFeature
 * Feature describing the ratio of white pixels present in a binary image
 * Note: Image/Cell must only contain 1 channel and be thersholded to 0 | 255
 */
class BinaryCountFeature : public FeatureFilter {
public:

    cv::Mat pixelRatios; // Stores the ratio of white to non-white pixels in each cell/image as the percentage of white pixels
    double pixelWeigth;
    int pixelCount;
    bool singleCell;

    /*
     * BinaryCountFeature (Constructor)
     * @param singleCell    - If true the feature is an average of all cells otherwise unique feature stored for every image cell
     * @param imageRows     - The number of rows in the image matrix
     * @param imageColumns  - The number of columns in the image matrix
     */
    BinaryCountFeature(bool singleCell, int imageRows, int imageColumns) {

        this->singleCell = singleCell;
        this->rowMargin = imageRows % cellSize;
        this->colMargin = imageColumns % cellSize;

        if (singleCell) {
            
            this->pixelCount = (imageRows - rowMargin) * (imageColumns - colMargin);
            this->pixelRatios = cv::Mat(imageRows, imageColumns, CV_64FC2);

        } else {
            this->pixelCount = cellSize * 2;
            this->pixelRatios = cv::Mat(1, 1, CV_64FC2);
        }

        this->pixelWeigth = 100 / this->pixelCount;
    }

    /*
     * BinaryCountFeature (Constructor)
     * @param singleCell    - If true the feature is an average of all cells otherwise unique feature stored for every image cell
     */
    BinaryCountFeature(bool singleCell) {

        this->singleCell = singleCell;
    }

    /*
     * initalize
     * Initalize the feature based on image dimensions 
     * @param imageRows     - The number of rows in the image matrix
     * @param imageColumns  - The number of columns in the image matrix
     */
    void initalize(int imageRows, int imageColumns) {
        this->rowMargin = imageRows % cellSize;
        this->colMargin = imageColumns % cellSize;

        if (singleCell) {
            
            this->pixelCount = (imageRows - rowMargin) * (imageColumns - colMargin);
            this->pixelRatios = cv::Mat(imageRows, imageColumns, CV_64FC2);

        } else {
            this->pixelCount = cellSize * 2;
            this->pixelRatios = cv::Mat(1, 1, CV_64FC2);
        }

        this->pixelWeigth = 100 / this->pixelCount;
    }

    /*
     * extractFeature
     * Extracts the ratio of white pixels in the passed (image/image cells) as a percentage of 100
     * @param image - The image that the white pixels will be counted in
     */
    void extractFeature(cv::Mat& image) override {

        int colIndex, rowIndex = 0; 
        for (int row = (rowMargin / 2); (row + cellSize) < image.rows - (rowMargin / 2); row += cellSize) {
            colIndex = 0;
            for (int col = colMargin / 2; (col  + cellSize) < image.cols - (colMargin / 2); col += cellSize) {

                cv::Mat cell = image(cv::Range(row, row + cellSize), cv::Range(col, col + cellSize));

                if (singleCell)
                    updatePixelValue(cell, pixelRatios.at<uchar>(0, 0));
                else
                    updatePixelValue(cell, pixelRatios.at<uchar>(rowIndex, colIndex));

                cell.release();
                colIndex++;
            }
            rowIndex++;
        }
    }

    /*
     * updateFeature
     * Updates the current feature to incorporate the pixel values in the passed image
     * @param image - The image providing pixel values for feature update
     */
    void updateFeature(cv::Mat& image) override {

        cv::Mat pixelRatios;

        // Update current feature values
        if (singleCell)
            pixelCount += (image.rows - rowMargin) * (image.cols - colMargin);
        else
            pixelCount *= 2; // cellSize is equal regardless of image size threfore * 2

        double oldPixelWeigth = pixelWeigth;
        pixelWeigth = 100.0 / pixelCount;

        double scale = oldPixelWeigth / pixelWeigth;

        for (int row = 0; row < pixelRatios.rows; row++)
            for (int col = 0; col < pixelRatios.cols; col++)
                pixelRatios.at<double>(row, col) *= scale;

        // Append new pixel values to feature values
        this->extractFeature(image);
    }

    /*
     * compare
     * Returns the difference between the ratio in white pixels. The return value
     * is positive if the ratio of the passed feature is greater else negative
     *
     * @parma feature The feature for comparision
     */
    double compare(FeatureFilter* feature) override {

        if (typeid(*feature) != typeid(BinaryCountFeature))
            std::cout << "Error invalid feature class (" << typeid(feature).name() << ") is an invalid type for comparsion operation\n";

        BinaryCountFeature* internalFeature = dynamic_cast<BinaryCountFeature*>(feature);

        if (singleCell) {
            return internalFeature->pixelRatios.at<double>(0,0) - pixelRatios.at<double>(0, 0);

        } else {
            double difference = 0;
            for (int row = 0; row < pixelRatios.rows; row++)
                for (int col = 0; col < pixelRatios.cols; col++)
                    difference += internalFeature->pixelRatios.at<double>(0,0) - pixelRatios.at<double>(0, 0);

            return difference;
        }
    }

    /*
     * updatePixelValue
     * Updates a feature value based on the current pixelValue
     * @param featureValue - Cell providing pixel values for feature update
     * @param featureValue - The value of the current feature
     */
    void updatePixelValue(cv::Mat& cell, unsigned char& featureValue) {

        for (int row = 0; row < cell.rows; row++) {
            for (int col = 0; col < cell.cols; col++) {

                if (cell.at<uchar>(row, col) == 255)
                     featureValue += pixelWeigth;
            }
        }
    }

};

/*
 * BinaryDistributionFeature
 * Describes the distribution of binary pixel values in an image
 * Note: Image must be of only 1 channel with values between 0 and 255 inclusive 
 */
class BinaryDistributionFeature : public FeatureFilter {

    void extractFeature(cv::Mat& cell) {
    }
    void updateFeature(cv::Mat& cell) {
    }
    double compare(FeatureFilter* feature) {

        //Cell evaluation
        /*
        float* normal = normalSample.ptr<float>(rowIndex,colIndex);
        float normalDistance = 0; float anomolyDistance = 0;
        for (int i = 0; i < 5; i++) {
            normalDistance += std::abs(cellLBPHistogram[i] - normal[i]);
            anomolyDistance += std::abs(cellLBPHistogram[i] - anomolySample[i]);
            //std::cout << "- normalSample[" << i << "]: " << normal[i] << "\n";
            //std::cout << "- anomalySample[" << i << "]: " << anomolySample[i] << "\n";
        }
        */

        // Mark anomoly
        //if (anomolyDistance < normalDistance) {
        /*
        if (whitePixelCount > 100) {
            std::cout << "\nanomaly\n";
            RGB colour = RGB{0,0,255};
            markFault(returnImage, col, col + global::cellSize, row , row + global::cellSize, nullptr, colour);
        } else
            std::cout << "\nnormal\n";
        std::cout << "anomalyDistance : " << anomolyDistance << "\nnormalDistance: " << normalDistance << "\n";
        */

        return 0;
    }

};

#endif
