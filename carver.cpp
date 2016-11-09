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
    cout << "Usage: " << argv[0] << " input-image new-x-dimension new-y-dimension output-image" << endl;
    return -1;
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
  } else if (xdelta > image.size().width - 2 || ydelta > image.size().height - 2) {
    cout << "target image size too small." << endl;
    return -1;
  } else {
    cout << "Remove " << xdelta << " columns and " << ydelta << " rows from the image." << endl;
  }

  // alternate removing horizontal/vertical seams until
  // target image size has been reached
  while(xdelta + ydelta  > 0) {
    if(xdelta > 0) {
      xdelta--;
      // first calculate energy of image. this is the the simple first derivative
      // as described in the paper
      Mat energy = calcEnergy(image);
      // calculate the resulting cost matrix using dynamic programming
      Mat cost = calcCost(energy, VERT);
      // uncomment these lines to see intermediate results for energy & cost matrix
      //  saveImageNormalized("cost.jpg", cost);
      //  saveImageNormalized("energy.jpg", energy);

      // find the cheapest path along this cost matrix
      vector<int> seam = findSeam(cost, VERT);
      // now remove the pixels on this seam and continue working with the smaller
      // image.
      removeSeam(image, seam, VERT).copyTo(image);
    }
    if(ydelta > 0) {
      ydelta--;
      Mat energy = calcEnergy(image);
      Mat cost = calcCost(energy, HORI);
      vector<int> seam = findSeam(cost, HORI);
      removeSeam(image, seam, HORI).copyTo(image);
    }
  }
  cout << "Saving new image as " << argv[4] << endl;
  imwrite(argv[4], image);
  cout << "All done." << endl;
  return 0;
}

/**
 *  Helper: save image with values normalized to 0-255 range
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
  // safeguard for max = 0, which would result in divbyzero
  if(max < 0.000000001) max = 0.000000001;
  alpha = 256.0/max;
  beta = -1.0 * min * (256.0/max);
  cout << "alpha " << alpha << " beta " << beta << endl;
  cout << image.size().width << image.size().height << endl;
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

      // use absolutes because values can be negative
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
  // code redundancy vs. speed - further comments apply to both directions.
  if(dir == HORI) {
    // cost for the first step is only their energy, so copy first column
    energy.col(0).copyTo(cost.col(0));

    imax = energy.size().width;
    jmax = energy.size().height;
    for (int i = 1; i < imax; i++) {
      for (int j = 0; j < jmax; j++) {
        l = max(0, j - 1);
        r = min(jmax - 1, j + 1);

        // find cheapest path so far by comparing left/middle/right
        float cheapest = min(min(cost.at<float>(l, i - 1), cost.at<float>(j, i - 1)),
                             cost.at<float>(r, i - 1));
        // cost is the total previous cost from the cheapest path so far and
        // the pixels own energy.
        cost.at<float>(j, i) = cheapest + energy.at<float>(j, i);
      }
    }
  } else {
    energy.row(0).copyTo(cost.row(0));
    imax = energy.size().height;
    jmax = energy.size().width;
    for (int i = 1; i < imax; i++) {
      for (int j = 0; j < jmax; j++) {
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
  // code redundancy vs. speed - further comments apply to both directions.
  if(dir == HORI) {
    // reserve final length of seam to avoid reallocations
    seam.reserve(cost.size().width - 1);

    // find starting point by looking for end point with least cost
    for(int y = 0; y < cost.size().height; y++) {
      if(cost.at<float>(y, cost.size().width - 1) < min) {
        min = cost.at<float>(y, cost.size().width - 1);
        min_pos = y;
      }
    }
    // save first step of seam
    seam.push_back(min_pos);

    // trace back the cheapest path from this point.
    int cur = min_pos;
    int next = cur;
    for(int x = cost.size().width - 1; x > 0; x--) {
      int above = cur;
      int below = cur;
      // watch out for edges!
      if(cur > 0) above--;
      if(cur < cost.size().height) below++;

      // select cheapest neighbour from next step above/middle/below.
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
      // copy unchanged pixels (between origin and seam) over to new image
      for (int j = 0; j < *it; j++) {
        output.at<Vec3b>(i, j) = input.at<Vec3b>(i, j);
      }
      // copy changed pixels (after seam) over, but shift by 1 in respective direction
      for( int j = *it + 1; j < output.size().width; j++) {
        output.at<Vec3b>(i, j - 1) = input.at<Vec3b>(i, j);
      }
      it++;
    }
  } else {
    for(int i = input.size().width - 1; i >= 0 && it != seam.end(); i--) {
      for (int j = 0; j < *it; j++) {
        output.at<Vec3b>(j, i) = input.at<Vec3b>(j, i);
      }
      for( int j = *it + 1; j < output.size().height; j++) {
        output.at<Vec3b>(j - 1, i) = input.at<Vec3b>(j, i);
      }
      it++;
    }
  }
  return output;
}
