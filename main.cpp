#include "main.hpp"

using namespace std;
using namespace cv;

int main( int argc, char** argv ) {
  if( argc != 5) {
    cout << "Usage: " << argv[0] << " [inputimage] [x-Delta] [y-Delta] [output]" << endl;
    return -1;
  }

  Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  if(!image.data) {
    cout << "Could not open or find the image" << endl;
    return -1;
  }

  Mat jpeg;
  int xdelta = atoi(argv[2]);
  int ydelta = atoi(argv[3]);
  while(xdelta + ydelta  > 0) {
    cout << xdelta << " " << ydelta << endl;
    Mat energy = calcEnergy(image);
    energy.convertTo(jpeg, CV_8UC1, 0.5);
    if(xdelta > 0) {
      xdelta--;
      Mat cost = calcCost(energy, VERT);
      cost.convertTo(jpeg, CV_8UC1, 1.0/32.0);
      imwrite("energy.jpg", jpeg);
      vector<int> seam = findSeam(cost, VERT);
      // debugging code: paint seam red and save image
       vector<int>::iterator it = seam.begin();
      for (int i = image.size().width; it != seam.end() && i > 0 ; ++it, i--) {
        image.at<Vec3b>(i - 1, *it) = Vec3b (0, 0, 255);
      }
    }
    if(ydelta > 0) {
      ydelta--;
      Mat cost = calcCost(energy, HORI);
      cost.convertTo(jpeg, CV_8UC1, 1.0/32.0);
      imwrite("energy.jpg", jpeg);
      vector<int> seam = findSeam(cost, HORI);
      // debugging code: paint seam red and save image
       vector<int>::iterator it = seam.begin();
      for (int i = image.size().width; it != seam.end() && i > 0 ; ++it, i--) {
        image.at<Vec3b>(*it, i -1) = Vec3b (0, 0, 255);
      }
    }
  }
  imwrite(argv[4], image);
    imwrite("cost.jpg", jpeg);
  //
  return 0;
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
      int xnext = min(xmax, x + 1);
      int yprev = max(0, y - 1);
      int ynext = min (ymax, y + 1);

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
  } else {
    energy.row(0).copyTo(cost.row(0));
    imax = energy.size().height;
    jmax = energy.size().width;
  }
  for (int i = 1; i < imax; i++) {
    for (int j = 0; j < jmax; j++) {
      // watch out for those borders.
      l = max(0, j - 1);
      r = min(jmax, j + 1);
      if(dir == HORI) {
        float cheapest = min(min(cost.at<float>(l, i - 1), cost.at<float>(j, i - 1)),
                             cost.at<float>(r, i - 1));
        // cost is the total previous cost from the cheapest path so far and the pixels own energy.
        cost.at<float>(j, i) = cheapest + energy.at<float>(j, i);
      } else {
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
    seam.reserve(cost.size().width);
    // find starting point by looking for last path point with least cost
    for(int y = 0; y < cost.size().height; y++) {
      if(cost.at<float>(y, cost.size().width) < min) {
        min = cost.at<float>(y, cost.size().width);
        min_pos = y;
      }
    }
    // traverse from this point back along the cheapest neighbours, memorize path
    seam.push_back(min_pos);
    int cur = min_pos;
    int next = cur;
    //TODO: check boundaries.
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
      if(cost.at<float>(cost.size().height, x) < min) {
        min = cost.at<float>(cost.size().height, x);
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
      if(cur < cost.size().width) right++;
      if(cost.at<float>(y - 1, left) < cost.at<float>(y - 1, cur)) next = left;
      if(cost.at<float>(y - 1, right) < cost.at<float>(y - 1, cur)) next = right;
      seam.push_back(next);
      cur = next;
    }
  }
  return seam;
}
