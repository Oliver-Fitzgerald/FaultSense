#include <opencv2/opencv.hpp>

int main() {
    cv::Mat img = cv::imread("../../../../../../data/chewinggum/Data/Images/Anomaly/004.JPG");
    if (img.empty()) return -1;

    cv::Rect roi = cv::selectROI("Select ROI", img);

    if (roi.width > 0 && roi.height > 0) {
        cv::Mat selected = img(roi);
        cv::imshow("Selected Area", selected);
    }

    return 0;
}
