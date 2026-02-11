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
#include "../../feature/pre-processing.h"
#include "../../feature/utils/pre-processing-utils.h"
#include "../../evaluation/evaluation.h"
#include "../../training/train.h"
#include "../../general/file-operations/training-data.h"
#include "../../general/file-operations/generic-read-write.h"

void view(std::string imagePath, std::map<std::string, bool> flags, unsigned int &noiseThreshold);
void evaluation(std::map<std::string, bool> flags);
void train(std::map<std::string, bool> flags);

int main(int argc, char** argv) {

    CLI::App faultSense{"An compter vision application for anomoly detection"};
    faultSense.require_subcommand();
    argv = faultSense.ensure_utf8(argv);

    // View subcommand
    CLI::App* viewSubcommand = faultSense.add_subcommand("view", "View image with optional filters applied")->ignore_case();
    std::string imagePath = "";
    unsigned int noiseThreshold = 0;
    std::map<std::string, bool> viewFlags = { {"markFault", false}, {"objectDetection", false}, {"lbp", false}, {"edge", false}, {"hsv", false}, {"removeNoise", false}};

    viewSubcommand->add_option("-i, --image", imagePath, "The path to an image");
    viewSubcommand->add_option("-r, --removeNoise", noiseThreshold, "removes any noise from the final image of size >= 0");
    viewSubcommand->add_flag("-o, --objectDetection", viewFlags["objectDetection"], "Applies object detection");
    viewSubcommand->add_flag("-l, --lbp", viewFlags["lbp"], "Applies local binary pattern to each pixel");
    viewSubcommand->add_flag("-e, --edge", viewFlags["edge"], "Applies canny edge detection to image");
    viewSubcommand->add_flag("--hsv", viewFlags["hsv"], "Applies a hue, staturation and value threshold on image"); // -h already in use for --help
    viewSubcommand->add_flag("-m, --markFault", viewFlags["markFault"], "Does fault detection and marks each of the predicted faulty celss in the final image");

    viewSubcommand->final_callback([&imagePath, &viewFlags, &noiseThreshold]() {
        if (imagePath == "")
            imagePath = "../data/chewinggum/Data/Images/Anomaly/000.JPG";

        if (imagePath.rfind("/") == imagePath.size() - 1) {
            std::map<std::string, cv::Mat> images = readImagesFromDirectory(imagePath);

            for (const auto& [imageName, image] : images)
                view(imagePath + imageName, viewFlags, noiseThreshold);

        } else
            view(imagePath, viewFlags, noiseThreshold);
    });

    // Evaluation subcommand
    CLI::App* evaluationSubcommand = faultSense.add_subcommand("eval", "Evaluates the trained norms")->ignore_case();
    std::map<std::string, bool> evalFlags = {{"chewinggum", false}};
    evaluationSubcommand->add_flag("--chewinggum", evalFlags["chewinggum"], "Evaluates the effectivness of trained norm at binary classification");

    evaluationSubcommand->final_callback([&evalFlags]() {
        evaluation(evalFlags);
    });

    // Train subcommand
    CLI::App* trainSubcommand = faultSense.add_subcommand("train", "Trains norms and writes result to file")->ignore_case();
    std::map<std::string, bool> trainFlags = {{"", false}};

    trainSubcommand->final_callback([&trainFlags]() {
        train(trainFlags);
    });

    CLI11_PARSE(faultSense, argc, argv);
    return 0;
}

/*
 * view
 * Displays an image given it's path, applying any pre-processing
 * techniques specified
 
 * @param imagePath The path to the image to be displayed
 * @param flags A list of flags to indicate which pre-processing techniques should be applied.
 */
std::map<std::string, std::array<float, 5>> normalNorm;
std::map<std::string, std::array<float, 5>> anomalyNorm;
void view(std::string imagePath, std::map<std::string, bool> flags, unsigned int &noiseThreshold) {

    cv::Mat temp = cv::imread(imagePath);
    cv::Mat original = cv::imread(imagePath);
    cv::Mat image;

    if (flags["objectDetection"])
        objectDetection(temp, image);
    else 
        image = temp.clone();

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

    if (noiseThreshold > 0) {
        removeNoise(image, noiseThreshold);
    }


    if (flags["markFault"]) {
        std::cout << "Generate normal norm cell\n";
        //std::map<std::string, std::array<float, 5>> normalNorm;
        if (normalNorm.size() == 0)
            trainCellNorms(normalNorm, true, "chewinggum");

        std::cout << "Generate anomaly norm cell\n";
        //std::map<std::string, std::array<float, 5>> anomalyNorm;
        if (anomalyNorm.size() == 0)
            trainCellNorms(anomalyNorm, false, "chewinggum");

        markFaultLBP(normalNorm["chewinggum"], anomalyNorm["chewinggum"], image);
        cv::imshow("Image", image);
    } else
        cv::imshow("Image", image);

    while (cv::pollKey() != 113);

}

/*
 * evaluation
 * Under Construction
 */
void evaluation(std::map<std::string, bool> flags) {

    std::cout << "Reading normal norm ...\n";
    std::map<std::string, cv::Mat> normalNorm = {
        {"chewinggum", cv::Mat()},
        {"candle", cv::Mat()},
        {"capsules", cv::Mat()},
        {"cashew", cv::Mat()},
        {"fryum", cv::Mat()},
        {"macaroni1", cv::Mat()},
        {"macaroni2", cv::Mat()},
        {"pcb1", cv::Mat()},
        {"pcb2", cv::Mat()},
        {"pcb3", cv::Mat()},
        {"pcb4", cv::Mat()},
        {"pipe_fryum", cv::Mat()}
    };
    readMatrixNorm(normalNorm);

    std::cout << "Reading anomaly norm ...\n";
    std::map<std::string, std::array<float, 5>> anomalyNorm;
    readCellDistributions(anomalyNorm);

    if (flags["chewinggum"]) {
        std::cout << "Evaluating chewing gum instances ...\n";
        evaluateNormal("chewinggum", normalNorm["chewinggum"], anomalyNorm["chewinggum"]);
    } else {
        std::cout << "Warning: No object type selected\n";
    }
}

/*
 * train
 * Under Construction
 */
void train(std::map<std::string, bool> flags) {

    std::cout << "Generate nomral norm matrix\n";
    std::map<std::string, cv::Mat> normalNorm;
    trainMatrix(normalNorm);
    std::cout << "Write normal norm to file\n";
    writeMatrixNorm(normalNorm); 

    std::cout << "Generate anomaly norm cell\n";
    std::map<std::string, std::array<float, 5>> anomalyNorm;
    trainCellNorms(anomalyNorm, false);
    std::cout << "Write anomaly norm to file\n";
    writeCellDistributions(anomalyNorm);

}
