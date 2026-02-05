/*
 * synthetic-data-generation
 */

// OpenCV2
#include <opencv2/opencv.hpp>
// Fault Sense
#include "../general/file-operations/generic-read-write.h"

void generateRemoveNoiseTestData();
void onMouse(int event, int, int, int, void*);

/*
 * generateRemoveNoiseTestData
 * Genereates synthetic images used for the testing of removeNoise function
 */
void generateRemoveNoiseTestData() {

    // Check if data has already been generated

    /*
     * 4x4 center group (no action)
     * wwww
     * wbbw
     * wbbw
     * wwww
     */
    cv::Mat image = cv::Mat(4,4, CV_8UC1);
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {

            if (row == 0 || row == 3 || col == 3 || col == 0)
                image.at<uchar>(row, col) = 0;
            else
                image.at<uchar>(row, col) = 255;
        }
    }

    cv::namedWindow("img", cv::WINDOW_NORMAL);
    cv::setMouseCallback("img", onMouse);
    cv::imshow("img", image);
    cv::waitKey(0);
    //while (cv::pollKey() != 113);
}

double scale = 1.0;
cv::Mat src;

void onMouse(int event, int, int, int, void*) {
    if (event == cv::EVENT_MOUSEWHEEL) {
        std::cout << "update\n";
        int delta = cv::getMouseWheelDelta(event);
        scale *= (delta > 0) ? 1.1 : 0.9;

        cv::Mat resized;
        cv::resize(src, resized, {}, scale, scale);
        cv::imshow("img", resized);
    } else 
        std::cout << "event: " << event << "\n";
}
