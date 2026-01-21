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

void view(std::string imagePath, std::map<std::string, bool> flags);

int main(int argc, char** argv) {

    CLI::App faultSense{"An compter vision application for anomoly detection"};
    faultSense.require_subcommand();
    argv = faultSense.ensure_utf8(argv);

    // View subcommand
    CLI::App* viewSubcommand = faultSense.add_subcommand("view", "View image with optional filters applied")->ignore_case();
    std::string imagePath = "";
    std::map<std::string, bool> viewFlags = {{"objectDetection", false}, {"lbp", false}};

    viewSubcommand->add_option("-i, --image", imagePath, "The path to an image")->required();
    viewSubcommand->add_flag("--objectDetection", viewFlags["objectDetection"], "Applies object detection");
    viewSubcommand->add_flag("--lbp", viewFlags["lbp"], "Applies local binary pattern to each pixel");

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

    if (flags["lbp"]) {
        temp = image;
        lbpValues(temp, image);
    }

    cv::imshow("Image", image);
    while (cv::pollKey() != 113);
}
