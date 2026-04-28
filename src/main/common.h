#pragma once

bool classify(const float& normalDistance, const float& anomalyDistance, const std::string& category, const int index);
void applyPreProcessing(cv::Mat& cell, const std::string& imageCategory, const int index);
bool isNormal(cv::Mat& cell);
int closestCommonFactor(int a, int b, int target);
void binaryMaskExtraction(cv::Mat& img16, cv::Mat& img8);

