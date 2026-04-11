#ifndef image_viewer_H
#define image_viewer_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <map>
#include <string>
// Fault Sense
#include "../../../objects/PreProcessingPipeline.h"

void imageViewer(cv::Mat &originalImage);
void view(cv::Mat &image, PreProcessingPipeline &preProcessingPipeline, std::map<std::string, bool>& viewFlags);

#endif
