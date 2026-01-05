/*
 * fault-sense-cli.cc
 * The main script for using fault sense
 */

int main(int argc, char** argv) {

    bool cannyObjectDetection = true; 
    HSV HSVThreshold{0, 22, 0, 119, 88,255};
    cv::Mat erodeKernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
    CannyThreshold cannyThreshold{116,0};

    std::vector<cv::Mat> images = readImagesFromDirectory("../data/chewinggum/Data/Images/Anomaly/");
    std::vector<cv::Mat> finalImages;

    std::cout << std::size(images) << "\n";
    //for (int i = 0; i < std::size(images); i++) {
    for (int i = 0; i < 1; i++) {

        std::cout << "Object " << i + 1 << " Detection: ";
        //cv::Mat img,image = images[i];
        cv::Mat image = cv::imread("../data/sample-images/chewinggum-anomoly.JPG");
        cv::Mat img = cv::imread("../data/sample-images/chewinggum-anomoly.JPG");
        if (cannyObjectDetection)
            edgeDetection(image, erodeKernal, cannyThreshold);
        else
            thresholdHSV(image, HSVThreshold);

        removeNoise(image);
        std::cout << ".";
        objectCoordinates coordinates = getEdges(img);
        markFault(image, coordinates.yMin, coordinates.yMax, coordinates.xMin, coordinates.xMax, "Object");
        std::cout << ".\n";

        finalImages.push_back(image);
        finalImages.push_back(img);
    }
        
    std::cout << "Object Detection Complete\n";

    // Display each image in a separate window
    for (int i = 0; i < finalImages.size(); i++) {
        std::string windowName;
        
        windowName = "Image " + std::to_string(i + 1);
        
        // Create and show window
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        cv::imshow(windowName, finalImages[i]);
        
        // Position windows in a cascading pattern (optional)
        //moveWindow(windowName, 50 + (i * 50), 50 + (i * 50));
    }
    
    std::cout << "Displaying " << images.size() << " images in separate windows." << std::endl;
    std::cout << "Press any key to close all windows..." << std::endl;
    while (true) cv::pollKey();
}

