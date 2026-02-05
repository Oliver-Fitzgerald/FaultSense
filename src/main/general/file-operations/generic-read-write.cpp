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

std::map<std::string, cv::Mat> readImagesFromDirectory(const std::string& directory);

/*
 * readImagesFromDirectory
 */
std::map<std::string, cv::Mat> readImagesFromDirectory(const std::string& directory) {

    namespace fs = std::filesystem;
    std::map<std::string, cv::Mat> images;
    
    try {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {

                std::string path = entry.path().string();
                std::string ext = entry.path().extension().string();
                
                // Check for common image extensions
                if (ext == ".JPG" || ext == ".jpeg" || ext == ".png" || 
                    ext == ".bmp" || ext == ".tiff" || ext == ".tif") {
                    
                    cv::Mat img = cv::imread(path);
                    if (!img.empty()) {
                        images.insert({path.substr(path.size() - 7), img});
                    } else {
                        std::cerr << "Failed to load: " << path << std::endl;
                    }
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
    
    return images;
}
