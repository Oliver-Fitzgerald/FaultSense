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
#include "image-viewer/image-viewer.h"
#include "evaluation/evaluation-interface.h"
#include "training/training-interface.h"
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

namespace {
    void readConfig(PreProcessingPipeline& preProcessingPipeline);
}

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




namespace {

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

}
