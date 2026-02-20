/*
 * image-viewer
 * This file provides functions for viewing an image and image transformation
 * tools including zooming
 */

// OpenCV2
#include <opencv2/opencv.hpp>
// Fault Sense
#include "../../../general/device_info.h"

namespace {

    double zoomFactor = 1.0;
    bool zoom(int key, cv::Mat &image, cv::Mat &result);
}

/*
 * imageViewer
 * The main function of the image viewer
 *
 * @param image The image to be displayed and transformed
 */
void imageViewer(cv::Mat &image) {

    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
    while (true) {

        int key = cv::waitKey(30);
        if (key == 113) break;

        int returnImageRows = image.rows * (zoomFactor * 1.5);
        int returnImageCols = image.cols * (zoomFactor * 1.5);
        if ( key == keys::PLUS && (returnImageRows > device::WINDOWHEIGHT || returnImageCols > device::WINDOWWIDTH))
            continue;
        returnImageRows = image.rows * (zoomFactor * 0.5);
        returnImageCols = image.cols * (zoomFactor * 0.5);
        if ( key == keys::MINUS && (returnImageRows < image.rows || returnImageCols < image.cols))
            continue;

        cv::Mat returnImage;
        if (!zoom(key, image, returnImage)) continue; // continue if invalid key


        cv::Mat canvas(device::WINDOWHEIGHT, device::WINDOWWIDTH, returnImage.type(), cv::Scalar(128));
        int x = (device::WINDOWWIDTH - returnImage.cols) / 2;
        int y = (device::WINDOWHEIGHT - returnImage.rows) / 2;

        returnImage.copyTo(canvas(cv::Rect(x, y, returnImage.cols, returnImage.rows)));
        cv::imshow("img", canvas);
    }
    cv::destroyAllWindows();
}

namespace {
    /*
     * zoom
     * Zooms in or out on an image
     *
     * @parm key the key pressed 
     * @param image the image to be zoomed in or out of
     * @param retult the image zoomed in or out
     */
    bool zoom(int key, cv::Mat &image, cv::Mat &result) {

        if (key == keys::PLUS) {
            zoomFactor *= 1.5;
        } else if (key == keys::MINUS) {
            zoomFactor *= 0.5;
        } else 
            return false;

        cv::resize(image, result, cv::Size(), zoomFactor, zoomFactor, cv::INTER_NEAREST);
        return true;
    }
}
