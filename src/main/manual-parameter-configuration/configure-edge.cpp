// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// Standard
#include <iostream>

int dilateX = 3, dilateY = 3;
int erodeX = 3, erodeY = 3;
int cannyX = 0, cannyY = 0;
int randA = 0, randB = 0;

/*
 * onKernalSizeChange
 * Ensures that the user input is always an odd number
 */
void onKernalSizeChange(int value, void* userdata) {

    const char* trackbar = static_cast<const char*>(userdata);

    if (value % 2 == 0) {
        value += 1;
        cv::setTrackbarPos(trackbar, "Trackbars", value);
    }
}


cv::Mat adaptive_exposure(const cv::Mat& img, float strength = 2.0, float curve = 2.0) {
    cv::Mat result = img.clone();
    
    for(int y = 0; y < img.rows; y++) {
        for(int x = 0; x < img.cols; x++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
            
            // Calculate intensity
            int intensity = (pixel[0] + pixel[1] + pixel[2]) / 3;
            float normalized = intensity / 255.0f;
            
            // Power curve: higher curve value = more weight to darks
            float boost = 1.0 + strength * pow(1.0 - normalized, curve);
            
            for(int c = 0; c < 3; c++) {
                result.at<cv::Vec3b>(y, x)[c] = 
                    cv::saturate_cast<uchar>(pixel[c] * boost);
            }
        }
    }
    return result;
}

// Usage:
// adaptive_exposure(img, 2.0, 2.0);  // Moderate boost to darks
// adaptive_exposure(img, 3.0, 3.0);  // Strong boost to darks
// adaptive_exposure(img, 5.0, 4.0);  // Very aggressive dark boost

int main(int argc, char** argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    if (argc != 2) {
        std::cerr << "Invalid Usage\nCorrect Usage: " << *argv << " <image_path>" << std::endl;
        return -1;
    }

    argv++;
    std::string imagePath = *argv;
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr << "Error: Invalid image path: " << imagePath << std::endl;
        return -1;
    }

    cv::namedWindow("Trackbars", (640,200));
    cv::createTrackbar("Dilate X", "Trackbars", &dilateX, 33, onKernalSizeChange, (void*)"Dilate X");
    cv::createTrackbar("Dilate Y", "Trackbars", &dilateY, 33, onKernalSizeChange, (void*)"Dilate Y");
    cv::createTrackbar("Erode X", "Trackbars", &erodeX, 33, onKernalSizeChange, (void*)"Erode X");
    cv::createTrackbar("Erode Y", "Trackbars", &erodeY, 33, onKernalSizeChange, (void*)"Erode Y");
    cv::createTrackbar("Canny Threshold X", "Trackbars", &cannyX, 255);
    cv::createTrackbar("Canny Threshold Y", "Trackbars", &cannyY, 255);
    cv::createTrackbar("Random A", "Trackbars", &randA, 10);
    cv::createTrackbar("Random B", "Trackbars", &randB, 10);

    cv::Mat finalImage, tempImage, imageCanny, greyScale;
    bool next = true;
    while (next) {

        cv::cvtColor(image, greyScale, cv::COLOR_BGR2GRAY);
        cv::Mat exposed_img = adaptive_exposure(image,randA,randB);

        cv::Canny(exposed_img, imageCanny, cannyX, cannyY);
        cv::Mat dilateKernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilateX, dilateY));
        cv::Mat erodeKernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erodeX, erodeY));
        //cv::dilate(imageCanny, tempImage, dilateKernal);
        //cv::erode(tempImage, finalImage, erodeKernal);
        finalImage = imageCanny;

        cv::imshow("HSV Image", finalImage);
        cv::imshow("Image", image);
        cv::waitKey(1);
    }
}
