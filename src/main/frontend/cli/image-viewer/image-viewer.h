#ifndef image_viewer_H
#define image_viewer_H

// OpenCV2
#include <opencv2/opencv.hpp>
// Standard
#include <map>
#include <string>

void imageViewer(cv::Mat &image);
void view(cv::Mat &image, std::map<std::string, bool>& viewFlags);

#endif
