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

void view(std::string imagePath);

int main(int argc, char** argv) {

    CLI::App faultSense{"An compter vision application for anomoly detection"};
    faultSense.require_subcommand();
    argv = faultSense.ensure_utf8(argv);

    // View subcommand
    std::string imagePath = "";
    CLI::App* viewSubcommand = faultSense.add_subcommand("view", "View image with optional filters applied")->ignore_case();
    viewSubcommand->add_option("-i, --image", imagePath, "The path to an image")->required();
    viewSubcommand->final_callback([&imagePath]() {
                view(imagePath);
            });

    CLI11_PARSE(faultSense, argc, argv);
    return 0;
}

void view(std::string imagePath) {

    cv::Mat image = cv::imread(imagePath);
    cv::imshow("Image", image);
    while (cv::pollKey() != 113);
}
