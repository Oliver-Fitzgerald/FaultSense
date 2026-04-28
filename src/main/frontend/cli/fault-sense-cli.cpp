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
#include <chrono>
#include <map>
#include <exception>
// Fault Sense
#include "../../feature/object-detection.h"
#include "../../feature/feature-extraction.h"
#include "../../feature/pre-processing.h"
#include "../../feature/utils/pre-processing-utils.h"
#include "../../feature/utils/generic-utils.h"
#include "../../evaluation/evaluation.h"
#include "../../training/train.h"
#include "../../general/file-operations/training-data.h"
#include "../../general/file-operations/ground-truth.h"
#include "../../general/file-operations/generic-read-write.h"
#include "../../objects/PixelCoordinates.h"
#include "../../objects/ConfusionMatrix.h"
#include "../../common.h"

void view(std::string imagePath, std::map<std::string, bool> flags, unsigned int &noiseThreshold);
void view(std::string imagePath, std::map<std::string, bool> flags, unsigned int &noiseThreshold, std::unique_ptr<cv::Mat> inputImage);
void evaluation(std::map<std::string, bool> flags);
void train(std::map<std::string, bool> flags);
std::string getImageCategory(std::string& imagePath);
std::string getMaskPath(std::string& imagePath);
std::string to_png(std::string filename);
void binaryMaskExtraction(cv::Mat& img16, cv::Mat& img8);
std::string extractDigits(const std::string& s);
std::string transformFilename(const std::string& filename);

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

            // new implmentation of view & readImageFromDirectory
            for (auto& [imageName, image] : images) {
                std::cout << "imagePath: " << imagePath << imageName << "\n";
                view(imagePath + imageName, viewFlags, noiseThreshold, std::make_unique<cv::Mat>(image));
            }

        } else {
            view(imagePath, viewFlags, noiseThreshold, nullptr);
        }
    });

    // Evaluation subcommand
    CLI::App* evaluationSubcommand = faultSense.add_subcommand("eval", "Evaluates the trained norms")->ignore_case();
    std::map<std::string, bool> evalFlags = {{"chewinggum", false}, {"cashew", false}};
    evaluationSubcommand->add_flag("--chewinggum", evalFlags["chewinggum"], "Evaluates the effectivness of trained norm at binary classification");
    evaluationSubcommand->add_flag("--cashew", evalFlags["cashew"], "Evaluates the effectivness of trained norm at binary classification");

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
bool trained = false;
std::map<std::string, int> categoryFeatureCount = {{"chewinggum", 1}, {"cashew", 1}};
std::vector<std::array<float, 5>> normalNorm;
std::vector<std::array<float, 5>> anomalyNorm;
void view(std::string imagePath, std::map<std::string, bool> flags, unsigned int &noiseThreshold, std::unique_ptr<cv::Mat> inputImage) {

auto start = std::chrono::high_resolution_clock::now();
    namespace fs = std::filesystem;

    cv::Mat temp;
    cv::Mat original;

    if (inputImage == nullptr) {
        temp = cv::imread(imagePath);
        original = cv::imread(imagePath);
    } else {
        temp = inputImage->clone();
        original = inputImage->clone();
    }

    cv::Mat image;
    std::string imageCategory = getImageCategory(imagePath);
    std::cout << "imageCategory: " << imageCategory << "\n";
    ObjectCoordinates objectBounds;


    cv::Mat objectPreProcessing;
    if (flags["objectDetection"]) {
        objectPreProcessing = objectDetection(temp, image, imageCategory, objectBounds);
    }  else 
        image = temp.clone();

    if (flags["lbp"]) {
        throw std::runtime_error("lbp not implmented");
        //lbpValues(temp, image);

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


    cv::Mat imageMask;
    if (flags["markFault"]) {

        fs::path p(imagePath);
    
        std::string filename = p.stem().string() + ".png";
        fs::path current = p.parent_path(); // Anomaly
        current = current.parent_path();    // Images
        current = current.parent_path();    // Data
        current = current.parent_path();    // {category}
        std::string category = current.filename().string();
        
        // Build output path
        fs::path maskPath = fs::path("../data/masks") / category / filename;
        
        if (!trained) {



            std::cout << "INFO: Generating normal norm cell ...\n";
            normalNorm.resize(categoryFeatureCount[category]);
            trainCell(normalNorm, true, imageCategory);

            std::cout << "INFO: Generating anomaly norm cell ...\n";
            anomalyNorm.resize(categoryFeatureCount[category]);
            trainCell(anomalyNorm, false, imageCategory);
            trained = true;
        }

        cv::Mat mask = cv::imread(maskPath);
        crop(mask, objectBounds.xMin, objectBounds.xMax, objectBounds.yMin, objectBounds.yMax, imageMask);

        // while (cv::pollKey() != 113) {
        //     cv::imshow("blank", image);
        // }
        //
        // int threshold = 0; int busyThreshold = 0;
        // cv::namedWindow("Trackbars", (640,200));
        // cv::createTrackbar("Busythreshold", "Trackbars", &busyThreshold, 10000);
        // cv::createTrackbar("normalthreshold", "Trackbars", &threshold, 10000);
        //
        // applyPreProcessing(image, imageCategory, 1);
        //
        // while (cv::pollKey() != 113) {
        //     cv::Mat atemp = image.clone();
        //     removeBusyNoise(atemp, busyThreshold);
        //     removeNoise(atemp, threshold);
        //     cv::imshow("blank", atemp);
        // }

        markFaultLBP(normalNorm, anomalyNorm, image, imageCategory, imageMask);
    }

    cv::Mat preProcessing = image.clone();
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
std::cout << "Time: " << duration.count() << " ms\n";
    applyPreProcessing(preProcessing, "cashew", 1); while (cv::pollKey() != 113) {
        cv::imshow("Showing an image", image);
        if (flags["markFault"])
            cv::imshow("Showing an image1", imageMask);
        if (flags["objectDetection"])
            cv::imshow("Showing an imagetemp", objectPreProcessing);
    }

}

/*
 * evaluation
 * Under Construction
 */
void evaluation(std::map<std::string, bool> flags) {


    std::string category;
    if (flags["chewinggum"]) {
        category =  "chewinggum";
    } else if (flags["cashew"]) {
        category = "cashew";
    } else
        throw std::invalid_argument("You must select an object category to be evaluated");

    if (!trained) {
        std::cout << "INFO: Generating normal norm cell ...\n";
        normalNorm.resize(categoryFeatureCount[category]);
        trainCell(normalNorm, true, category);

        std::cout << "INFO: Generating anomaly norm cell ...\n";
        anomalyNorm.resize(categoryFeatureCount[category]);
        trainCell(anomalyNorm, false, category);
        trained = true;
    }

    std::string imagePath = "../data/" + category + "/Data/Images";
    std::map<std::string, cv::Mat> images;

    ConfusionMatrix confusionMatrix = ConfusionMatrix{0,0,0,0};
    ConfusionMatrix localizationConfusionMatrix = ConfusionMatrix{0,0,0,0};

    std::cout << "INFO: Evaluating normal images ... \n";
    images = readImagesFromDirectory(imagePath + "/Normal/");
    for (auto& [imageName, image] : images) {
        bool result = evaluate(localizationConfusionMatrix, category, normalNorm, anomalyNorm, image);
        confusionMatrix.update(result, true);
    }

    std::cout << "INFO: Evaluating anomaly images ... \n";
    images = readImagesFromDirectory(imagePath + "/Anomaly/");

    std::map<std::string, std::array<std::string, 5>> objectLabels;
    readVisaLabels(category, objectLabels);

    for (auto& [imageName, image] : images) {

        // bool hit = false;
        // for (auto& faultType : objectLabels[transformFilename(imageName)]) {
        //     if (faultType == "burnt") hit = true;
        // }
        // if (!hit) continue;
        // std::cout << "imageName: " << imageName << "\n";

        cv::Mat imageMask = cv::imread("../data/masks/" + category + "/" + to_png(imageName));
        cv::Mat binaryMask;
        binaryMaskExtraction(imageMask, binaryMask);
        bool result = evaluate(localizationConfusionMatrix, category, normalNorm, anomalyNorm, image, binaryMask);
        confusionMatrix.update(result, false);
    }

    std::cout << "Localization Confusion Matrix\n";
    std::cout << localizationConfusionMatrix << std::endl;
    std::cout << "Confusion Matrix\n";
    std::cout << confusionMatrix << std::endl;
}

/*
 * train
 * Under Construction
 */
void train(std::map<std::string, bool> flags) {

    throw std::runtime_error("The train functionality has not yet been implmented");
    // std::cout << "Generate nomral norm matrix\n";
    // std::map<std::string, cv::Mat> normalNorm;
    // trainMatrix(normalNorm);
    // std::cout << "Write normal norm to file\n";
    // writeMatrixNorm(normalNorm); 
    //
    // std::cout << "Generate anomaly norm cell\n";
    // std::map<std::string, std::array<float, 5>> anomalyNorm;
    // trainCell(anomalyNorm, false);
    // std::cout << "Write anomaly norm to file\n";
    // writeCellDistributions(anomalyNorm);

}


void view(std::string imagePath, std::map<std::string, bool> flags, unsigned int &noiseThreshold) {
    view(imagePath, flags, noiseThreshold, nullptr);
}

std::string getImageCategory(std::string& imagePath) {

    size_t firstSlash  = imagePath.find('/');
    size_t secondSlash = imagePath.find('/', firstSlash + 1);
    size_t thirdSlash  = imagePath.find('/', secondSlash + 1);
    
    return imagePath.substr(secondSlash + 1, thirdSlash - secondSlash - 1);
 }

std::string getMaskPath(std::string& imagePath) {

    namespace fs = std::filesystem;

    fs::path p(imagePath);

    std::string filename = p.stem().string() + ".png";
    fs::path current = p.parent_path(); // Anomaly
    current = current.parent_path();    // Images
    current = current.parent_path();    // Data
    current = current.parent_path();    // {category}

    std::string category = current.filename().string();
    fs::path maskPath = fs::path("../data/masks") / category / filename;
    return maskPath.string();
}

std::string to_png(std::string filename) {
    std::size_t dotPos = filename.rfind('.');
    if (dotPos != std::string::npos) {
        filename.replace(dotPos, std::string::npos, ".png");
    }
    return filename;
}

std::string transformFilename(const std::string& filename) {
    std::string result = filename;

    // 1. Ensure extension is lowercase for easier matching
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // 2. Replace .jpg -> .png
    size_t pos = result.rfind(".jpg");
    if (pos != std::string::npos) {
        result.replace(pos, 4, ".png");
    }

    // 3. Restore original numeric casing assumption (optional skip if not needed)
    // (We assume only extension case mattered, so we proceed)

    // 4. Reconstruct with "A" prefix before filename (not path-safe version)
    result = "A" + result;

    return result;
}
