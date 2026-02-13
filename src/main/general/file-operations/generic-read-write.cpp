/*
 * generic-read-write
 * Contains generic read write operations
 */

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <string>
#include <filesystem>
#include <map>
#include <unordered_set>

namespace {
    bool isImageFile(std::filesystem::directory_entry imagePath);
}

/*
 * readImagesFromDirectory
 */
void readImagesFromDirectory(const std::string& directory, std::map<std::string, cv::Mat> &images) {

    namespace fs = std::filesystem;
    
    try {
        for (const auto& entry : fs::directory_iterator(directory)) {

            if (!entry.is_regular_file()) {
                std::cerr << "Skipping read of image [" << entry.path().string() << "] - Not regular file\n";
                continue;
            }
            if (!isImageFile(entry)) {
                std::cerr << "Skipping read of image [" << entry.path().string() << "] - Invalid type\n";
                continue;
            }


            std::string path = entry.path().string();
            std::string filename = path.substr(path.size() - 7);

            if (images.count(filename) > 0) {
                std::cerr << "Skipping read of image [" << path << "] - Duplicate filename\n";
            } else
                images.emplace(std::move(filename), cv::imread(path));
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
}


/*
 * readImagesFromDirectory
 */
void readImagesFromDirectory(const std::string& directory, std::vector<cv::Mat> &images) {

    namespace fs = std::filesystem;
    
    try {
        for (const auto& entry : fs::directory_iterator(directory)) {

            if (!entry.is_regular_file()) {
                std::cerr << "Skipping read of image [" << entry.path().string() << "] - Not regular file\n";
                continue;
            }
            if (!isImageFile(entry)) {
                std::cerr << "Skipping read of image [" << entry.path().string() << "] - Invalid type\n";
                continue;
            }
                
            images.push_back(cv::imread(entry.path().string()));
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
}    

namespace {

    /*
     * isImageFile
     * Validates wether a image file path terminates an image of a  valid image type
     * 
     * @param imagePath the file path pointing to the image file
     */
    bool isImageFile(std::filesystem::directory_entry imagePath) {

        static const std::unordered_set<std::string> validExtensions = { ".JPG" };
        std::string extension = imagePath.path().extension().string();
        
        return validExtensions.count(extension) > 0;
    }
}
