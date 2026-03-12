// Group Cells to from histogram
int length = (LBPValues.cols / 2) * (LBPValues.rows / 2);
int index = 0;
int grps[length][4] = {0};
for (int row = 0; row < LBPValues.rows; row++) {
    for (int col = 0; col < LBPValues.cols; col++) {

        int value = LBPValues.at<uint8_t>(row, col);
        std::cout << "dims: " << LBPValues.dims << "\n";

        if (row % 2 == 0 && col % 2 == 0) {
            grps[index][0] = value;
            index ++;
        } else if (row % 2 == 0 && col % 2 == 1)
            grps[index][1] = value;
        else if (row % 2 == 1 && col % 2 == 0)
            grps[index][2] = value;
        else if (row % 2 == 1 && col % 2 == 1)
            grps[index][3] = value;
    }
}


// Calculate difference between LBP cells
int values[length] = {0};
for (int i = 1; i < length; i++) {

    double distance = 0;
    for (int k = 0; k < 4; k++)
        distance += grps[i][k] - grps[i - 1][k];

    values[i - 1] = distance > 120 ? 255 : 0;

}

// Populate image of differnce between LBP cells
cv::Mat finalImage = cv::Mat::zeros(LBPValues.rows / 2, LBPValues.cols / 2, CV_8UC1);
index = 0;
for (int row = 0; row < LBPValues.rows / 2; row++) {
    for (int col = 0; col < LBPValues.cols / 2; col++) {
        //std::cout << "(row, col) => (" << row << ", " << col << ")\n";
        //std::cout << "value => " << values[index] << "\n";
        // std::cout << "index => " << index << "\n";
        finalImage.at<uint8_t>(row, col) = values[index];

        index++;
    }
}
