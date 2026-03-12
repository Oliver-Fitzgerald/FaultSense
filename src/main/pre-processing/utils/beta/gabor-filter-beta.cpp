#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/**
 * Apply Gabor filter to an image
 * 
 * @param src Input image (grayscale)
 * @param ksize Size of the Gabor kernel (must be odd)
 * @param sigma Standard deviation of the Gaussian envelope
 * @param theta Orientation of the normal to the parallel stripes (in radians)
 * @param lambda Wavelength of the sinusoidal factor
 * @param gamma Spatial aspect ratio
 * @param psi Phase offset
 * @return Filtered image
 */
Mat applyGaborFilter(const Mat& src, Size ksize, double sigma, double theta, 
                     double lambda, double gamma, double psi) {
    // Create Gabor kernel
    Mat kernel = getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, CV_32F);
    
    // Apply filter using filter2D
    Mat dst;
    filter2D(src, dst, CV_32F, kernel);
    
    return dst;
}

/**
 * Apply multiple Gabor filters with different orientations
 * 
 * @param src Input image (grayscale)
 * @param numOrientations Number of different orientations
 * @return Vector of filtered images
 */
vector<Mat> applyMultiOrientationGabor(const Mat& src, int numOrientations = 8) {
    vector<Mat> results;
    
    // Parameters for Gabor filter
    Size ksize(31, 31);
    double sigma = 5.0;
    double lambda = 10.0;
    double gamma = 0.5;
    double psi = 0;
    
    // Apply filters at different orientations
    for (int i = 0; i < numOrientations; i++) {
        double theta = CV_PI * i / numOrientations;
        Mat filtered = applyGaborFilter(src, ksize, sigma, theta, lambda, gamma, psi);
        results.push_back(filtered);
    }
    
    return results;
}

int main(int argc, char** argv) {
    // Load image
    string imagePath = (argc > 1) ? argv[1] : "input.jpg";
    Mat img = imread(imagePath, IMREAD_GRAYSCALE);
    
    if (img.empty()) {
        cout << "Error: Could not load image from " << imagePath << endl;
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }
    
    // Convert to float for better precision
    Mat imgFloat;
    img.convertTo(imgFloat, CV_32F, 1.0/255.0);
    
    // Example 1: Single Gabor filter
    cout << "Applying single Gabor filter..." << endl;
    Mat result = applyGaborFilter(imgFloat, Size(31, 31), 5.0, 0, 10.0, 0.5, 0);
    
    // Normalize for display
    Mat resultNorm;
    normalize(result, resultNorm, 0, 255, NORM_MINMAX);
    resultNorm.convertTo(resultNorm, CV_8U);
    
    // Display results
    namedWindow("Original", WINDOW_NORMAL);
    namedWindow("Gabor Filtered", WINDOW_NORMAL);
    imshow("Original", img);
    imshow("Gabor Filtered", resultNorm);
    
    // Example 2: Multiple orientations
    cout << "Applying multi-orientation Gabor filters..." << endl;
    vector<Mat> multiResults = applyMultiOrientationGabor(imgFloat, 8);
    
    // Display multiple orientations
    for (size_t i = 0; i < multiResults.size(); i++) {
        Mat temp;
        normalize(multiResults[i], temp, 0, 255, NORM_MINMAX);
        temp.convertTo(temp, CV_8U);
        
        string windowName = "Orientation " + to_string(i);
        namedWindow(windowName, WINDOW_NORMAL);
        imshow(windowName, temp);
    }
    
    // Save results
    imwrite("gabor_result.jpg", resultNorm);
    cout << "Result saved as 'gabor_result.jpg'" << endl;
    
    cout << "Press any key to exit..." << endl;
    waitKey(0);
    destroyAllWindows();
    
    return 0;
}
