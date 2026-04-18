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

    virtual void extractFeature(cv::Mat& cell) = 0;
    virtual void updateFeature(cv::Mat& cell) = 0;
    virtual int compareFeature (FeatureFilter* feature) = 0;

    virtual ~FeatureFilter() = default;
};


/*
 * BinaryCountFeature
 * Feature describing the ratio of white pixels present in a binary image
 * Note: Image/Cell must only contain 1 channel and be thersholded to 0 | 255
 */
class BinaryCountFeature : public FeatureFilter {
public:

    int pixelCount = 0;
    double whitePixelRatio = 0;

    /*
     * extractFeature
     * Extracts the ratio of white pixels in the passed image as a percentage of 100
     * @param image The image that the white pixels will be counted in
     */
    void extractFeature(cv::Mat& cell) override {

        pixelCount = cell.rows * cell.cols;
        int pixelWeigth = pixelCount / 100;

        for (int rows = 0; rows < cell.rows; rows++) {
            for (int cols = 0; cols < cell.cols; cols++) {
                
                int pixel = cell.at<uchar>(rows, cols);
                if (pixel == 255)
                    whitePixelRatio += pixelWeigth;

            }
        }
    }

    void updateFeature(cv::Mat& cell) override {
        pixelCount = cell.rows * cell.cols;
        int pixelWeigth = pixelCount / 100;

        for (int rows = 0; rows < cell.rows; rows++) {
            for (int cols = 0; cols < cell.cols; cols++) {
                
                int pixel = cell.at<uchar>(rows, cols);
                if (pixel == 255)
                    whitePixelRatio += pixelWeigth;

            }
        }
    }

    /*
     * compareFeature
     * Returns the difference between the ratio in white pixels. The return value
     * is positive if the ratio of the passed feature is greater else negative
     *
     * @parma feature The feature for comparision
     */
    int compareFeature (FeatureFilter* feature) override {

        if (typeid(feature).name() != "BinaryCountFeature")
            std::cout << "Error invalid feature class (" << typeid(feature).name() << ") is an invalid type\n";

        BinaryCountFeature* internalFeature = dynamic_cast<BinaryCountFeature*>(feature);
        if (internalFeature->whitePixelRatio > whitePixelRatio)
            return (int) internalFeature->whitePixelRatio - whitePixelRatio;
        else
            return (int) whitePixelRatio - internalFeature->whitePixelRatio;

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
    int compareFeature (FeatureFilter* feature) {

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

        return 0;
    }

};

#endif
