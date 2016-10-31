#ifndef MAIN_HPP
#define MAIN_HPP

#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define HORI 0
#define VERT 1

cv::Mat calcEnergy(cv::Mat input);
cv::Mat calcCost(cv::Mat input, int dir);
std::vector<int> findSeam(cv::Mat input, int dir);

#endif
