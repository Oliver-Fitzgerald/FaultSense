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
#include <cctype>
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
#include "../../objects/FeaturesCollection.h"

namespace {
    Mode modeFromString(std::string stringValue);
    std::unique_ptr<FeatureFilter> featureFromString(std::string stringValue);
    void readConfig(FeaturesCollection& features);
    void parseArray(const char* featuresValue);
}

int main(int argc, char** argv) {

    // Read Configuration
    FeaturesCollection features;
    readConfig(features);

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


    std::cout << "Fault Sense CLI\n";
    std::cout << "=====================\n";
    for (auto& [feature, preProcessingPipeline] : features.features) {
        std::cout << "Feature: " << typeid(feature.get()).name() << "\n";
        std::cout << preProcessingPipeline.get();
        std::cout << "=====================\n";
    }
    std::cout << "execution...\n";

    // View subcommand
    CLI::App* viewSubcommand = faultSense.add_subcommand("view", "View image with optional filters applied")->ignore_case();
    std::map<std::string, bool> viewFlags = { {"markFault", false}, {"getRegion", false} };
    viewSubcommand->add_flag("-m, --markFault", viewFlags["markFault"], "Does fault detection and marks each of the predicted faulty celss in the final image");
    viewSubcommand->add_flag("-r, --getRegion", viewFlags["getRegion"], "Gets a selected region of an image and writes is to a file");

    viewSubcommand->final_callback([&images, &viewFlags, &features] {
        for (auto& image : images) {
            view(image, features, viewFlags);
        }
    });


    // Evaluation subcommand
    CLI::App* evaluationSubcommand = faultSense.add_subcommand("eval", "Evaluates the trained norms")->ignore_case();
    std::map<std::string, bool> evalFlags = {{"chewinggum", false}};
    evaluationSubcommand->add_flag("--chewinggum", evalFlags["chewinggum"], "Evaluates the effectivness of trained norm at binary classification");

    evaluationSubcommand->final_callback([&evalFlags, &features]() {
        evaluation(evalFlags, features);
    });


    // Train subcommand
    CLI::App* trainSubcommand = faultSense.add_subcommand("train", "Trains norms and writes result to file")->ignore_case();
    std::map<std::string, bool> trainFlags = {{"", false}};

    trainSubcommand->final_callback([&trainFlags, &features]() {
        train(trainFlags, features);
    });

    CLI11_PARSE(faultSense, argc, argv);
    return 0;
}




namespace {

    /*
     * readConfig
     * Reads in the project configuration file to set the pre-processing techniques applied throughout the
     * project
     * @param features The features to be extracted, trained, evaluated arcoss system functions and the 
     *                 pre-processing configuration to be applied to extract the feature
     */
    void readConfig(FeaturesCollection& features) {

        CSimpleIniA ini;
        SI_Error rc = ini.LoadFile("../configuration.ini");
        if (rc < SI_OK) {
            std::cerr << "Error loading configuration file: " << rc << std::endl;
            return;
        }

        const std::string available_features[2] = {"BinaryCountFeature", "BinaryDistributionFeature"};

        for (auto featureName : available_features) {

            bool exists = ini.SectionExists(featureName.c_str());
            if (exists && ini.GetValue(featureName.c_str(), "enabled", "false") == "true") {

                std::unique_ptr<FeatureFilter> feature = std::make_unique<BinaryCountFeature>();
                std::unique_ptr<PreProcessingPipeline> preProcessingPipeline = std::make_unique<PreProcessingPipeline>();

                std::string section = featureName + ".ObjectDetection";
                exists = ini.SectionExists(section.c_str());
                if (exists && ini.GetValue(featureName.c_str(), "enabled", "false") == "true") {

                    PreProcessing objectDetection;
                    objectDetection.applyObjectDetection = true;
                    objectDetection.mode = modeFromString(ini.GetValue("ObjectDetection", "mode", "NONE"));
                    objectDetection.noiseThreshold = (int) ini.GetLongValue("ObjectDetection", "noiseThreshold");
                    preProcessingPipeline->objectDetectionConfiguration = objectDetection;
                }

                section = featureName + ".PreProcessing";
                exists = ini.SectionExists(section.c_str());
                if (exists && ini.GetValue(section.c_str(), "enabled", "false") == "true") {

                    PreProcessing preProcessing;
                    preProcessing.mode = modeFromString(ini.GetValue("PreProcessing", "mode", "NONE"));
                    preProcessing.noiseThreshold = (int) ini.GetLongValue("PreProcessing", "noiseThreshold");
                    preProcessingPipeline->preProcessingConfiguration = preProcessing;
                }

                features.features.emplace(std::move(feature), std::move(preProcessingPipeline));
            }
        }
    }

    /*
     * modeFromString
     * Converts string from configuration file to the relevant pre-processing
     * mode
     * @param preProcessingMode the string representation of the preProcessingMode
     */
    Mode modeFromString(std::string stringValue) {


        if (stringValue == "HSV" || stringValue == "hsv")
            return Mode::HSV;
        else if (stringValue == "EDGE" || stringValue == "edge")
            return Mode::EDGE;
        else if (stringValue == "LBP" || stringValue == "lbp")
            return Mode::LBP;
        else if (stringValue == "NONE" || stringValue == "none")
            return Mode::NONE;
        else
            throw std::invalid_argument("Error convertign string to pre-processing-mode: " + stringValue);
    }

    /*
     * featureFromString
     * Converts string from configuration file to the relevant object
     * @param stringValue the string representation of the feature classes name
     */
    std::unique_ptr<FeatureFilter> featureFromString(std::string stringValue) {

        if (stringValue == "BinaryCountFeature")
            return std::make_unique<BinaryCountFeature>();
        else if (stringValue == "BinaryDistributionFeature")
            return std::make_unique<BinaryDistributionFeature>();
        else
            throw std::invalid_argument("Error convertign string to feature class: " + stringValue);
    }

    /*
     * parseArray
     * Parses and ini array of format "key = [value1, value2, value3]"
     * an array value can be obtained with the following function call
     * ini.GetValue("Section", "key", "");
     * @param featureValue The array of values format "[value1, value2, value3]"
     */
    void parseArray(const char* featuresValue) {
        if (!featuresValue || strlen(featuresValue) <= 0)
            throw std::invalid_argument("[Features] not defined in configuration.ini\n");

        std::string featuresStr(featuresValue);
        
        // Strip surrounding brackets [ ]
        size_t start = featuresStr.find('[');
        size_t end = featuresStr.find(']');
        
        // Parse feature parameters
        if (start != std::string::npos && end != std::string::npos && end > start) {
            std::string inner = featuresStr.substr(start + 1, end - start - 1);
            
            std::stringstream ss(inner);
            std::string token;
            
            // Split by comma and trim whitespace
            while (std::getline(ss, token, ',')) {
                // Trim leading/trailing whitespace
                size_t tokenStart = token.find_first_not_of(" \t");
                size_t tokenEnd = token.find_last_not_of(" \t");
                
                if (tokenStart != std::string::npos) {
                    std::cout << "token: " << token.substr(tokenStart, tokenEnd - tokenStart + 1) << "\n";
                }
            }
        }
    }
}
