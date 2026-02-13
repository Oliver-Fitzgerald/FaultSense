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
#include "../../objects/PreProcessing.h"

void view(cv::Mat &image, PreProcessing &preProcessingConfig, PreProcessing &objectDetectionConfig, bool markFault);
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
    std::map<std::string, bool> viewFlags = { {"markFault", false}, {"lbp", false}, {"edge", false}, {"hsv", false}, {"removeNoise", false}};
    viewSubcommand->add_option("-i, --image", imagePath, "The path to an image");
    viewSubcommand->add_option("-r, --removeNoise", noiseThreshold, "removes any noise from the final image of size >= 0");
    viewSubcommand->add_flag("-l, --lbp", viewFlags["lbp"], "Applies local binary pattern to each pixel");
    viewSubcommand->add_flag("-e, --edge", viewFlags["edge"], "Applies canny edge detection to image");
    viewSubcommand->add_flag("--hsv", viewFlags["hsv"], "Applies a hue, staturation and value threshold on image"); // -h already in use for --help
    viewSubcommand->add_flag("-m, --markFault", viewFlags["markFault"], "Does fault detection and marks each of the predicted faulty celss in the final image");

    CLI::App* objectDetectionSubcommand = viewSubcommand->add_subcommand("objectDetection", "Applies object detection with the passed pre-processing flags applied")->ignore_case();
    unsigned int objectNoiseThreshold = 0;
    std::map<std::string, bool> objectDetectionFlags = { {"objectDetection", false}, {"edge", false}, {"hsv", false}, {"removeNoise", false}};
    objectDetectionSubcommand->add_flag("-e, --edge", objectDetectionFlags["edge"], "Applies canny edge detection to image");
    objectDetectionSubcommand->add_flag("--hsv", objectDetectionFlags["hsv"], "Applies a hue, staturation and value threshold on image"); // -h already in use for --help
    objectDetectionSubcommand->add_option("-r, --removeNoise", objectNoiseThreshold, "removes any noise from the final image of size >= 0");
    objectDetectionSubcommand->final_callback([&objectDetectionFlags]() {
        objectDetectionFlags["objectDetection"] = true;
    });
    viewSubcommand->final_callback([&imagePath, &viewFlags, &objectDetectionFlags, &noiseThreshold, &objectNoiseThreshold]() {

        // Configure PreProcessing flags
        PreProcessing preProcessingConfiguration;
        preProcessingConfiguration.lbp = viewFlags["lbp"];
        preProcessingConfiguration.hsv = viewFlags["hsv"];
        preProcessingConfiguration.edge = viewFlags["edge"];
        preProcessingConfiguration.noiseThreshold = noiseThreshold;
        PreProcessing objectDetectionConfiguration;
        objectDetectionConfiguration.hsv = objectDetectionFlags["hsv"];
        objectDetectionConfiguration.edge = objectDetectionFlags["edge"];
        objectDetectionConfiguration.enableObjectDetection = objectDetectionFlags["objectDetection"];
        objectDetectionConfiguration.noiseThreshold = objectNoiseThreshold;

        // Configure image path
        if (imagePath == "")
            imagePath = "../data/chewinggum/Data/Images/Anomaly/000.JPG";

        if (imagePath.rfind("/") == imagePath.size() - 1) {
            std::vector<cv::Mat> images;
            readImagesFromDirectory(imagePath, images);

            for (auto& image : images)
                view(image, preProcessingConfiguration, objectDetectionConfiguration, viewFlags["markFault"]);

        } else {
            cv::Mat image = cv::imread(imagePath);
            view(image, preProcessingConfiguration, objectDetectionConfiguration, viewFlags["markFault"]);
        }
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
std::map<std::string, std::array<float, 5>> normalNorm = {{"chewinggum", std::array<float, 5>()}};
std::map<std::string, cv::Mat> normalMatrixNorm = {{"chewinggum", cv::Mat()}};
std::map<std::string, std::array<float, 5>> anomalyNorm = {{"chewinggum", std::array<float, 5>()}};
bool first = true;
void view(cv::Mat &image, PreProcessing &preProcessingConfig, PreProcessing &objectDetectionConfig, bool markFault) {

    // Object Detection
    if (objectDetectionConfig.enableObjectDetection)
        objectDetectionConfig.apply(image);

    // Pre-Processing
    preProcessingConfig.apply(image);

    if (markFault) {

        std::cout << "Generate normal norm cell\n";
        //std::map<std::string, std::array<float, 5>> normalNorm; // Commented out temporarily so it's not re-calculated each time
        ///trainCellNorms(normalNorm,preProcessingConfig, true);
        if (first)
            trainMatrix(normalMatrixNorm, preProcessingConfig, true);


        std::cout << "Generate anomaly norm cell\n";
        //std::map<std::string, std::array<float, 5>> anomalyNorm; // Commented out temporarily so it's not re-calculated each time
        if (first)
            trainCellNorms(anomalyNorm, preProcessingConfig, false);

        if (first)
            first = false;
        markFaultLBP(normalMatrixNorm["chewinggum"], anomalyNorm["chewinggum"], image);
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

    PreProcessing objectDetectionConfiguration;
    objectDetectionConfiguration.hsv = true;
    objectDetectionConfiguration.noiseThreshold = 200;

    PreProcessing preProcessingConfiguration;
    preProcessingConfiguration.edge = true;


    std::cout << "Generate nomral norm matrix\n";
    std::map<std::string, cv::Mat> normalNorm;
    trainMatrix(normalNorm, preProcessingConfiguration, true);
    std::cout << "Write normal norm to file\n";

    std::cout << "Generate anomaly norm cell\n";
    std::map<std::string, std::array<float, 5>> anomalyNorm = {{"chewinggum", std::array<float,5>()}};
    trainCellNorms(anomalyNorm, preProcessingConfiguration, false);
    std::cout << "Write anomaly norm to file\n";
    writeCellDistributions(anomalyNorm);

}
