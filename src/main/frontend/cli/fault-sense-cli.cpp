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
// simpleini
#include "SimpleIni.h"
// Fault Sense
#include "image-viewer-ui/image-viewer.h"
#include "../../pre-processing/object-detection.h"
#include "../../pre-processing/pre-processing.h"
#include "../../pre-processing/utils/pre-processing-utils.h"
#include "../../feature/feature-extraction.h"
#include "../../evaluation/evaluation.h"
#include "../../training/train.h"
#include "../../general/file-operations/training-data.h"
#include "../../general/file-operations/generic-file-operations.h"
#include "../../objects/PreProcessing.h"
#include "../../objects/PreProcessingPipeline.h"
#include "../../objects/Features.h"

void view(cv::Mat &image, PreProcessingPipeline &preProcessingPipeline, std::map<std::string, bool>& viewFlags);
void evaluation(std::map<std::string, bool> flags, PreProcessingPipeline& preProcessingPipeline);
void train(std::map<std::string, bool> flags, PreProcessingPipeline& preProcessingPipeline);
void readConfig(PreProcessingPipeline& preProcessingPipeline);

int main(int argc, char** argv) {

    // Read Configuration
    PreProcessingPipeline preProcessingPipeline;
    readConfig(preProcessingPipeline);

    std::string imagePath = "";
    std::vector<cv::Mat> images;

    // Initalize CLI
    CLI::App faultSense{"An compter vision application for anomoly detection"};
    faultSense.require_subcommand();
    faultSense.add_option("-i, --image", imagePath, "The path to an image");
    argv = faultSense.ensure_utf8(argv);

    // Load Images
    faultSense.parse_complete_callback([&imagePath, &images] {
        if (imagePath == "")
            imagePath = "../data/chewinggum/Data/Images/Anomaly/000.JPG";

        if (imagePath.rfind("/") == imagePath.size() - 1) {
            readImagesFromDirectory(imagePath, images);

        } else {
            cv::Mat image = cv::imread(imagePath);
            images.push_back(image);
        }
    });


    // View subcommand
    CLI::App* viewSubcommand = faultSense.add_subcommand("view", "View image with optional filters applied")->ignore_case();
    std::map<std::string, bool> viewFlags = { {"markFault", false}, {"getRegion", false} };
    viewSubcommand->add_flag("-m, --markFault", viewFlags["markFault"], "Does fault detection and marks each of the predicted faulty celss in the final image");
    viewSubcommand->add_flag("-r, --getRegion", viewFlags["getRegion"], "Gets a selected region of an image and writes is to a file");

    viewSubcommand->final_callback([&images, &viewFlags, &preProcessingPipeline] {
        for (auto& image : images) {
            view(image, preProcessingPipeline, viewFlags);
        }
    });


    // Evaluation subcommand
    CLI::App* evaluationSubcommand = faultSense.add_subcommand("eval", "Evaluates the trained norms")->ignore_case();
    std::map<std::string, bool> evalFlags = {{"chewinggum", false}};
    evaluationSubcommand->add_flag("--chewinggum", evalFlags["chewinggum"], "Evaluates the effectivness of trained norm at binary classification");

    evaluationSubcommand->final_callback([&evalFlags, &preProcessingPipeline]() {
        evaluation(evalFlags, preProcessingPipeline);
    });


    // Train subcommand
    CLI::App* trainSubcommand = faultSense.add_subcommand("train", "Trains norms and writes result to file")->ignore_case();
    std::map<std::string, bool> trainFlags = {{"", false}};

    trainSubcommand->final_callback([&trainFlags, &preProcessingPipeline]() {
        train(trainFlags, preProcessingPipeline);
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
void view(cv::Mat& image, PreProcessingPipeline& preProcessingConfiguration, std::map<std::string, bool>& viewFlags) {



    if (viewFlags["markFault"]) {
        std::map<std::string, cv::Mat> normalMatrixNorm = {{"chewinggum", cv::Mat()}};
        readMatrixNorm(normalMatrixNorm);

        std::map<std::string, std::array<float, 5>> anomalyNorm = {{"chewinggum", std::array<float, 5>()}};
        readCellDistributions(anomalyNorm);

        FeatureFilter* param = new BinaryCountFeature();
        markFaultLBP(*param, preProcessingConfiguration, normalMatrixNorm["chewinggum"], anomalyNorm["chewinggum"], image);


    } else if (viewFlags["getRegion"]) {

        preProcessingConfiguration.apply(image);

        // Let user select ROI interactively
        cv::Rect roi = cv::selectROI("Select Region", image);

        // If width/height are zero, selection was cancelled
        if (roi.width == 0 || roi.height == 0) {
            std::cerr << "No region selected.\n";
        }

        // Crop the selected region
        cv::Mat cropped = image(roi).clone();  // clone to ensure independent copy

        // Save to file
        if (!cv::imwrite("output.jpg", cropped)) {
            std::cerr << "Error: Could not write output file.\n";
        }


    } else
        preProcessingConfiguration.apply(image);

    imageViewer(image);
}

/*
 * evaluation
 * Under Construction
 */
void evaluation(std::map<std::string, bool> flags, PreProcessingPipeline& preProcessingPipeline) {

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
    std::map<std::string, std::array<float, 5>> anomalyNorm = {{"chewinggum", std::array<float, 5>()}};
    readCellDistributions(anomalyNorm);


    if (flags["chewinggum"]) {
        std::cout << "Evaluating chewing gum instances ...\n";
        evaluateObjectCategory("chewinggum", normalNorm["chewinggum"], anomalyNorm["chewinggum"], preProcessingPipeline);
    } else {
        std::cout << "Warning: No object type selected\n";
    }
}

/*
 * train
 * Under Construction
 */
void train(std::map<std::string, bool> flags, PreProcessingPipeline& preProcessingPipeline) {

    std::cout << "Generate nomral norm matrix\n";
    std::map<std::string, cv::Mat> normalNorm = {{"chewinggum", cv::Mat()}};
    trainMatrix(normalNorm, preProcessingPipeline, true);
    std::cout << "Write normal norm to file\n";
    writeMatrixNorm(normalNorm); 

    std::cout << "Generate anomaly norm cell\n";
    std::map<std::string, std::array<float, 5>> anomalyNorm = {{"chewinggum", std::array<float,5>()}};
    trainCellNorms(anomalyNorm, preProcessingPipeline, false);
    std::cout << "Write anomaly norm to file\n";
    writeCellDistributions(anomalyNorm);

}

/*
 * readConfig
 * Reads in the project configuration file to set the pre-processing techniques applied throughout the
 * project
 */
void readConfig(PreProcessingPipeline& preProcessingPipeline) {

    CSimpleIniA ini;
    SI_Error rc = ini.LoadFile("../configuration.ini");
    if (rc < SI_OK) {
        std::cerr << "Error loading configuration file: " << rc << std::endl;
        return;
    }

    bool exists = ini.SectionExists("ObjectDetection");
    if (exists) {
        PreProcessing objectDetection;
        objectDetection.enableObjectDetection = true;
        objectDetection.lbp = ini.GetBoolValue("ObjectDetection", "lbp");
        objectDetection.hsv = ini.GetBoolValue("ObjectDetection", "hsv");
        objectDetection.edge = ini.GetBoolValue("ObjectDetection", "edge");
        objectDetection.noiseThreshold = (int) ini.GetLongValue("ObjectDetection", "noiseThreshold");
        preProcessingPipeline.steps.push_back(objectDetection);
    }

    exists = ini.SectionExists("PreProcessing");
    if (exists) {
        PreProcessing preProcessing;
        preProcessing.lbp = ini.GetBoolValue("PreProcessing", "lbp");
        preProcessing.hsv = ini.GetBoolValue("PreProcessing", "hsv");
        preProcessing.edge = ini.GetBoolValue("PreProcessing", "edge");
        preProcessing.noiseThreshold = (int) ini.GetLongValue("PreProcessing", "noiseThreshold");
        preProcessingPipeline.steps.push_back(preProcessing);
    }
}
