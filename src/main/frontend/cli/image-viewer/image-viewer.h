#ifndef image_viewer_H
#define image_viewer_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <map>
#include <string>
// Fault Sense
#include "../../../objects/FeaturesCollection.h"

void imageViewer(cv::Mat &image);
void view(cv::Mat& image, std::map<std::string, bool>& viewFlags, std::string& objectCategory, FeaturesCollection& features, std::string imageName);

#endif
