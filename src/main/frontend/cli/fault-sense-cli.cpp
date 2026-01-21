/*
 * fault-sense-cli.cc
 * The main script for using fault sense
 */

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// CLI11
#include <CLI/CLI.hpp>
// Standard
#include <map>
// Fault Sense
#include "../../feature/object-detection.h"
#include "../../feature/feature-extraction.h"
#include "../../feature/utils/pre-processing-utils.h"

void view(std::string imagePath, std::map<std::string, bool> flags);

int main(int argc, char** argv) {

    CLI::App faultSense{"An compter vision application for anomoly detection"};
    faultSense.require_subcommand();
    argv = faultSense.ensure_utf8(argv);

    // View subcommand
    CLI::App* viewSubcommand = faultSense.add_subcommand("view", "View image with optional filters applied")->ignore_case();
    std::string imagePath = "";
    std::map<std::string, bool> viewFlags = {{"objectDetection", false}, {"lbp", false}, {"edge", false}, {"hsv", false}};

    viewSubcommand->add_option("-i, --image", imagePath, "The path to an image")->required();
    viewSubcommand->add_flag("-o, --objectDetection", viewFlags["objectDetection"], "Applies object detection");
    viewSubcommand->add_flag("-l, --lbp", viewFlags["lbp"], "Applies local binary pattern to each pixel");
    viewSubcommand->add_flag("-e, --edge", viewFlags["edge"], "Applies canny edge detection to image");
    viewSubcommand->add_flag("--hsv", viewFlags["hsv"], "Applies a hue, staturation and value threshold on image"); // -h already in use for --help

    viewSubcommand->final_callback([&imagePath, &viewFlags]() {
        view(imagePath, viewFlags);
    });

    CLI11_PARSE(faultSense, argc, argv);
    return 0;
}

/*
 * view
 * Displays an image given it's path, applying any pre-processing
 * techniques specified
 *
 * @param imagePath The path to the image to be displayed
 * @param flags A list of flags to indicate which pre-processing techniques should be applied.
 */
void view(std::string imagePath, std::map<std::string, bool> flags) {

    cv::Mat temp = cv::imread(imagePath);
    cv::Mat image = temp;

    if (flags["objectDetection"]) {
        objectDetection(temp, image);
    }

    temp = image;
    if (flags["lbp"]) {
        lbpValues(temp, image);

    } else if (flags["edge"]) {
        CannyThreshold threshold{57, 29};
        cv::Mat kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
        edgeDetection(image, kernal, threshold);

    } else if (flags["hsv"]) {
        HSV HSVThreshold{0, 22, 0, 119, 88,255};
        thresholdHSV(image, HSVThreshold);
    } 

    
    cv::imshow("Image", image);
    while (cv::pollKey() != 113);
}
