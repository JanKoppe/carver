/* carver.hpp -- seam carving implementation done the hard way.
 *
 * Copyright (C) 2016 Jan Koppe
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE file for details.
 */

#ifndef CARVER_HPP
#define CARVER_HPP

#include <cmath>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define HORI 0
#define VERT 1

cv::Mat calcEnergy(cv::Mat);
cv::Mat calcCost(cv::Mat, int);
std::vector<int> findSeam(cv::Mat, int);
cv::Mat removeSeam(cv::Mat, std::vector<int>, int);
void saveImageNormalized(const std::string& name, cv::Mat image);

#endif
