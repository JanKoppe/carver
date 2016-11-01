#ifndef MAIN_HPP
#define MAIN_HPP

#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define HORI 0
#define VERT 1

cv::Mat calcEnergy(cv::Mat);
cv::Mat calcCost(cv::Mat, int);
std::vector<int> findSeam(cv::Mat, int);
cv::Mat removeSeam(cv::Mat, std::vector<int>, int);

#endif
