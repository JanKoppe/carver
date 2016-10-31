#include "main.hpp"

int main( int argc, char** argv ) {
  if( argc != 5) {
    std::cout << "Usage: " << argv[0] << " [inputimage] [x-Delta] [y-Delta] [output]" << std::endl;
    return -1;
  }

  cv::Mat image;
  image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

  if(!image.data) {
    std::cout << "Could not open or find the image" << std::endl;
    return -1;
  }

  std::cout << "calculate energy function" << std::endl;
  cv::Mat energy = calcEnergy(image);

  cv::Mat jpeg;
  energy.convertTo(jpeg, CV_8UC1, 0.5);
  cv::imwrite("energy.jpg", jpeg);

  std::cout << "calculate cost matrix" << std::endl;
  cv::Mat cost = calcCost(energy, HORI);

  cost.convertTo(jpeg, CV_8UC1, 0.2);
  cv::imwrite("cost.jpg", jpeg);

  std::vector<int> seam = findSeam(cost, HORI);

  // debugging code: paint seam red and save image
   std::vector<int>::iterator it = seam.begin();
  for (int x = image.size().width; it != seam.end() && x > 0 ; ++it, x++) {
    image.at<cv::Vec3b>(*it, x) = cv::Vec3b (0, 0, 255);
  }
  cv::imwrite(argv[4], image);
  //
  return 0;
}

/**
 * Calculate energy values for pixels through their horizontal and
 * vertical gradients.
 * @param input Matrix to calculate the energy for.
 * @return Matrix containing the energy values for pixels in input.
 */
cv::Mat calcEnergy(cv::Mat input) {
  cv::Mat energy = cv::Mat::zeros(input.size(), CV_32FC1);
  for(int x = 0; x < energy.size().width; x++) {
    for(int y = 0; y < energy.size().height; y++) {
      int xPrev = x;
      int xNext = x;
      int yPrev = y;
      int yNext = y;
      // prevent attempts to access outside of image
      if(x > 0) xPrev--;
      if(x < energy.size().width) xNext++;
      if(y > 0) yPrev--;
      if(y < energy.size().height) yNext++;

      // Calculate gradient energy for a pixel as described on chapter 4, slide 11.
      int xEnergy =
        (input.at<cv::Vec3b>(y, xPrev).val[0] - input.at<cv::Vec3b>(y, xNext).val[0]) / 2 +
        (input.at<cv::Vec3b>(y, xPrev).val[1] - input.at<cv::Vec3b>(y, xNext).val[1]) / 2 +
        (input.at<cv::Vec3b>(y, xPrev).val[2] - input.at<cv::Vec3b>(y, xNext).val[2]) / 2 ;
      int yEnergy =
        (input.at<cv::Vec3b>(yPrev, x).val[0] - input.at<cv::Vec3b>(yPrev, x).val[0]) / 2 +
        (input.at<cv::Vec3b>(yPrev, x).val[1] - input.at<cv::Vec3b>(yPrev, x).val[1]) / 2 +
        (input.at<cv::Vec3b>(yPrev, x).val[2] - input.at<cv::Vec3b>(yPrev, x).val[2]) / 2 ;

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
cv::Mat calcCost(cv::Mat energy, int dir) {
  cv::Mat cost = cv::Mat::zeros(energy.size(), energy.type());
  if(dir == HORI) {
    // cost for the first step is their own energy.
    energy.col(0).copyTo(cost.col(0));
    for (int x = 1; x < energy.size().width; x++) {
      for (int y = 0; y < energy.size().height; y++) {
        int above = y;
        int below = y;
        // watch out for those borders.
        if (y > 0) above--;
        if (y < energy.size().height) below++;

        float cheapest = std::min(std::min(cost.at<float>(above, x - 1), cost.at<float>(y, x - 1)),
                            cost.at<float>(below, x - 1));
        // cost is the total previous cost from the cheapest path so far and the pixels own energy.
        cost.at<float>(y, x) = cheapest + energy.at<float>(y, x);
      }
    }
  } else {
    energy.row(0).copyTo(cost.row(0));
    for (int y = 1; y < energy.size().width; y++) {
      for (int x = 0; x < energy.size().height; x++) {
        int left = x;
        int right = x;
        if (x > 0) left--;
        if (x < energy.size().width) right++;

        float pixelcost = std::min(std::min(cost.at<float>(y - 1, left), cost.at<float>(y - 1, x)),
                            cost.at<float>(y - 1, right));
        cost.at<float>(y, x) = pixelcost + energy.at<float>(y, x);
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
std::vector<int> findSeam(cv::Mat cost, int dir) {
  std::vector<int> seam;
  int min_pos;
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
