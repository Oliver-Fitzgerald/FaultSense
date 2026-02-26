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
    void showImage(cv::Mat &image, cv::Mat &canvas);
}

/*
 * imageViewer
 * The main function of the image viewer
 *
 * @param image The image to be displayed and transformed
 */
void imageViewer(cv::Mat &originalImage) {

    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
    cv::Mat canvas(device::WINDOWHEIGHT, device::WINDOWWIDTH, originalImage.type(), cv::Scalar(128));

    cv::Mat image;
    double scaleX = static_cast<double>(device::WINDOWWIDTH  - 10) / originalImage.cols;
    double scaleY = static_cast<double>(device::WINDOWHEIGHT - 10) / originalImage.rows;
    double scale  = std::min(scaleX, scaleY);

    cv::resize(originalImage, image, cv::Size(), scale, scale, cv::INTER_AREA);
    showImage(canvas, image);

    while (true) {
        canvas = cv::Mat(device::WINDOWHEIGHT, device::WINDOWWIDTH, image.type(), cv::Scalar(128));

        int key = cv::waitKey(30);
        if (key == 113) break;

        int returnImageRows = image.rows * (zoomFactor * 1.5);
        int returnImageCols = image.cols * (zoomFactor * 1.5);
        if ( key == keys::PLUS && (returnImageRows > device::WINDOWHEIGHT || returnImageCols > device::WINDOWWIDTH)) {
            showImage(canvas, image);
            continue;
        }

        cv::Mat returnImage;
        if (!zoom(key, image, returnImage)) continue; // continue if invalid key

        showImage(canvas, returnImage);
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


    /*
     * showImage
     * @parm canvas the canvas the image will be displayed on
     * @param image the image to be shown
     */
    void showImage(cv::Mat &canvas, cv::Mat &image) {
                                          //
        int x = (device::WINDOWWIDTH - image.cols) / 2;
        int y = (device::WINDOWHEIGHT - image.rows) / 2;

        image.copyTo(canvas(cv::Rect(x, y, image.cols, image.rows)));
        cv::imshow("img", canvas);
    }
}
