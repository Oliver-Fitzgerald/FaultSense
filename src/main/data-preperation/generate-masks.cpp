/*
 * generate-masks
 * This file hands the logic for thresholding the masks from the raw dataset
 * to make them visible in an image viewer, as well as compressing there size.
 */

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Standard
#include <array>
#include <iostream>
#include <map>
#include <exception>
// Fault Sense
#include "../feature/utils/generic-utils.h"

int main (int argc, char** argv) {

    std::string dataSetRoot = "../data/";
    std::array<std::string, 12> categories = {
        "pcb1/",
        "pcb2/",
        "pcb3/",
        "pcb4/",
        "capsules/",
        "candle/",
        "macaroni1/",
        "macaroni2/",
        "cashew/",
        "chewinggum/",
        "fryum/",
        "pipe_fryum/"
    };

    for (int index = 0; index < 12; index++) {
        std::string path = dataSetRoot + categories[index] + "Data/Masks/Anomaly/";
        std::map<std::string, cv::Mat> rawMasks = readImagesFromDirectory(path);

        // Generate masks
        for (auto const& [name, image] : rawMasks) {

            std::string maskPath;
            try {
                // Threshold rough mask
                cv::Mat greyScale, finalMask;
                cv::cvtColor(image, greyScale, cv::COLOR_BGR2GRAY);
                cv::threshold(greyScale, finalMask, 0,255,cv::THRESH_BINARY);

                if (finalMask.empty()) throw std::domain_error("finalMask is empty");

                // Write to appropriate path
                maskPath = dataSetRoot + "masks/" + categories[index] + name;
                cv::imwrite(maskPath, finalMask);

            } catch(const std::domain_error exception) {
                fprintf(stderr, "There was an issue thresholding the mask of %s: %s\n", path + name, exception.what());

            } catch (const cv::Exception& exception) {
                fprintf(stderr, "Exception writing image %s: %s\n", maskPath, exception.what());
            }

        }
    }
    return 0;
}
