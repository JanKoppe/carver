/* carver.cpp -- seam carving implementation done the hard way.
 *
 * Copyright (C) 2016 Jan Koppe
 *
 * This software may be modified and distributed under the terms
 * of the MIT license.  See the LICENSE file for details.
 */

#include "carver.hpp"

using namespace std;
using namespace cv;

int main( int argc, char** argv ) {
  if(argc != 5) {
    cout << "wrong arguments. usage: carver input x-dimension y-dimension output." << endl;
  }

  Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

  if(!image.data) {
    cout << "Could not open or find the image" << endl;
    return -1;
  }

  // calculate delta from actual and target image size
  int xdelta = image.size().width - atoi(argv[2]);
  int ydelta = image.size().height - atoi(argv[3]);

  if(xdelta < 0 || ydelta < 0) {
    cout << "can only shrink images. target size bigger than original." << endl;
    return -1;
  }

  // alternate removing horizontal/vertical seams until
  // target image size has been reached
  while(xdelta + ydelta  > 0) {
    cout << "xdelta " << xdelta << " ydelta " << ydelta << endl;
    if(xdelta > 0) {
      xdelta--;
      Mat energy = calcEnergy(image);
      Mat cost = calcCost(energy, VERT);
      vector<int> seam = findSeam(cost, VERT);
      removeSeam(image, seam, VERT).copyTo(image);
      cout << "removed one vertical seam." << endl;
    }
    if(ydelta > 0) {
      ydelta--;
      Mat energy = calcEnergy(image);
      Mat cost = calcCost(energy, HORI);
      saveImageNormalized("cost.jpg", cost);
      saveImageNormalized("energy.jpg", energy);
      vector<int> seam = findSeam(cost, HORI);
      removeSeam(image, seam, HORI).copyTo(image);
      cout << "removed one horizontal seam." << endl;
    }
  }
  imwrite(argv[4], image);
  return 0;
}

/**
 *  Helper function: save normalized image
 *  @param name
 *  @param image
 *  @return void
 */
void saveImageNormalized(const std::string& name, cv::Mat image) {
  double min, max;
  Mat jpeg;
  //find minimum and maximum values in mat to properly resize to 8 bits
  minMaxLoc(image, &min, &max);
  double alpha, beta;
  alpha = 256.0/max;
  beta = -1.0 * min * (256.0/max);
  cout << "alpha " << alpha << " beta " << beta << endl;
  image.convertTo(jpeg, CV_8UC3, alpha, beta);
  imwrite(name, jpeg);
  return;
}

/**
 * Calculate energy values for pixels through their horizontal and
 * vertical gradients.
 * @param input Matrix to calculate the energy for.
 * @return Matrix containing the energy values for pixels in input.
 */
Mat calcEnergy(Mat input) {
  Mat energy = Mat::zeros(input.size(), CV_32FC1);
  int xmax = energy.size().width;
  int ymax = energy.size().height;
  for(int x = 0; x < xmax; x++) {
    for(int y = 0; y < ymax; y++) {
      // prevent attempts to access outside of image
      int xprev = max(0, x - 1);
      int xnext = min(xmax - 1, x + 1);
      int yprev = max(0, y - 1);
      int ynext = min (ymax - 1, y + 1);

      // Calculate gradient energy for a pixel as described on chapter 4, slide 11.
      int xEnergy =
        (input.at<Vec3b>(y, xprev).val[0] - input.at<Vec3b>(y, xnext).val[0]) / 2 +
        (input.at<Vec3b>(y, xprev).val[1] - input.at<Vec3b>(y, xnext).val[1]) / 2 +
        (input.at<Vec3b>(y, xprev).val[2] - input.at<Vec3b>(y, xnext).val[2]) / 2 ;
      int yEnergy =
        (input.at<Vec3b>(yprev, x).val[0] - input.at<Vec3b>(ynext, x).val[0]) / 2 +
        (input.at<Vec3b>(yprev, x).val[1] - input.at<Vec3b>(ynext, x).val[1]) / 2 +
        (input.at<Vec3b>(yprev, x).val[2] - input.at<Vec3b>(ynext, x).val[2]) / 2 ;

      // use absolute values because energy can be negative
      // see avidan et al., section 3 for reference
      energy.at<float>(y, x) = abs(xEnergy) + abs(yEnergy);
    }
  }
  return energy;
}

/**
 * Calculate the path cost matrix for a given energy matrix with dynamic programming.
 * @param energy Matrix with energy values for each pixel
 * @param dir Sets horizontal (ltr) or vertical (ttb) direction.
 * @return Path cost matrix of same type as energy matrix.
 */
Mat calcCost(Mat energy, int dir) {
  Mat cost = Mat::zeros(energy.size(), energy.type());
  int imax, jmax, l, r;
  if(dir == HORI) {
    // cost for the first step is their own energy.
    energy.col(0).copyTo(cost.col(0));
    imax = energy.size().width;
    jmax = energy.size().height;
    for (int i = 1; i < imax; i++) {
      for (int j = 0; j < jmax; j++) {
        // watch out for those borders.
        l = max(0, j - 1);
        r = min(jmax - 1, j + 1);
        float cheapest = min(min(cost.at<float>(l, i - 1), cost.at<float>(j, i - 1)),
                             cost.at<float>(r, i - 1));
        // cost is the total previous cost from the cheapest path so far and the pixels own energy.
        cost.at<float>(j, i) = cheapest + energy.at<float>(j, i);
      }
    }
  } else {
    energy.row(0).copyTo(cost.row(0));
    imax = energy.size().height;
    jmax = energy.size().width;
    for (int i = 1; i < imax; i++) {
      for (int j = 0; j < jmax; j++) {
        // watch out for those borders.
        l = max(0, j - 1);
        r = min(jmax - 1, j + 1);
        float cheapest = min(min(cost.at<float>(i - 1, l), cost.at<float>(i - 1, j)),
                             cost.at<float>(i - 1, r));
        cost.at<float>(i, j) = cheapest + energy.at<float>(i, j);
      }
    }
  }
  return cost;
}

/**
 * Find the cheapest seam in the cost matrix.
 * @param cost Matrix
 * @param dir Direction of the cost matrix and seam
 * @return vector of seamcoordinates in respect to the direction
 */
vector<int> findSeam(Mat cost, int dir) {
  vector<int> seam;
  int min_pos, imax, jmax;
  float min = FLT_MAX;
  if(dir == HORI) {
    seam.reserve(cost.size().width - 1);
    // find starting point by looking for end point with least cost
    for(int y = 0; y < cost.size().height; y++) {
      if(cost.at<float>(y, cost.size().width - 1) < min) {
        min = cost.at<float>(y, cost.size().width - 1);
        min_pos = y;
      }
    }
    seam.push_back(min_pos);
    // trace back cheapest path and save steps
    int cur = min_pos;
    int next = cur;
    for(int x = cost.size().width - 1; x > 0; x--) {
      int above = cur;
      int below = cur;
      // watch out for those pesky edges!
      if(cur > 0) above--;
      if(cur < cost.size().height) below++;
      // select cheapest neighbour
      if(cost.at<float>(above, x - 1) < cost.at<float>(cur, x - 1)) next = above;
      if(cost.at<float>(below, x - 1) < cost.at<float>(cur, x - 1)) next = below;
      // memorize, continue.
      seam.push_back(next);
      cur = next;
    }
  } else {
    seam.reserve(cost.size().height);
    for(int x = 0; x < cost.size().width; x++) {
      if(cost.at<float>(cost.size().height - 1, x) < min) {
        min = cost.at<float>(cost.size().height - 1, x);
        min_pos = x;
      }
    }
    seam.push_back(min_pos);
    int cur = min_pos;
    int next = cur;
    for(int y = cost.size().height - 1; y > 0; y--) {
      int left = cur;
      int right = cur;
      if(cur > 0) left--;
      if(cur < cost.size().width - 1) right++;
      if(cost.at<float>(y - 1, left) < cost.at<float>(y - 1, cur)) next = left;
      if(cost.at<float>(y - 1, right) < cost.at<float>(y - 1, cur)) next = right;
      seam.push_back(next);
      cur = next;
    }
  }
  return seam;
}

/**
 * Remove a Seam from the input Image in direction dir
 * @param input Image
 * @param seam  vector for seam
 * @param dir   direction of the seam
 * @return  image with seam removed
 */
cv::Mat removeSeam(cv::Mat input, std::vector<int> seam, int dir) {
  cv::Mat output = Mat::zeros(Size(input.size().width - dir, input.size().height - (dir ^ 1)), input.type());
  vector<int>::iterator it = seam.begin();
  if(dir == VERT) {
    for(int i = input.size().height - 1; i >= 0 && it != seam.end(); i--) {
      // unchanged pixels
      for (int j = 0; j < *it; j++) {
        output.at<Vec3b>(i, j) = input.at<Vec3b>(i, j);
      }
      // shift pixels after seam
      for( int j = *it; j < output.size().width; j++) {
        output.at<Vec3b>(i, j) = input.at<Vec3b>(i, j + 1);
      }
      it++;
    }
  } else {
    for(int i = input.size().width - 1; i >= 0 && it != seam.end(); i--) {
      for (int j = 0; j < *it; j++) {
        output.at<Vec3b>(j, i) = input.at<Vec3b>(j, i);
      }
      for( int j = *it; j < output.size().height; j++) {
        output.at<Vec3b>(j, i) = input.at<Vec3b>(j + 1, i);
      }
      it++;
    }
  }
  return output;
}
