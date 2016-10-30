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

  std::cout << "calculate cost matrix" << std::endl;
  cv::Mat cost = calcCost(energy, HORI);

  return 0;
}

cv::Mat calcEnergy(cv::Mat input) {
  cv::Mat output = cv::Mat::zeros(input.size(), CV_8UC1);
  for(int x = 0; x < output.size().width; x++) {
    for(int y = 0; y < output.size().height; y++) {
      int xPrev = x;
      int xNext = x;
      int yPrev = y;
      int yNext = y;
      // prevent attempts to access outside of image
      if(x > 0) xPrev--;
      if(x < output.size().width) xNext++;
      if(y > 0) yPrev--;
      if(y < output.size().height) yNext++;

      // Calculate gradient energy for a pixel as described on chapter 4, slide 11.
      int xEnergy =
        (input.at<cv::Vec3b>(xPrev, y).val[0] - input.at<cv::Vec3b>(xNext, y).val[0]) / 2 +
        (input.at<cv::Vec3b>(xPrev, y).val[1] - input.at<cv::Vec3b>(xNext, y).val[1]) / 2 +
        (input.at<cv::Vec3b>(xPrev, y).val[2] - input.at<cv::Vec3b>(xNext, y).val[2]) / 2 ;
      int yEnergy =
        (input.at<cv::Vec3b>(x, yPrev).val[0] - input.at<cv::Vec3b>(x, yPrev).val[0]) / 2 +
        (input.at<cv::Vec3b>(x, yPrev).val[1] - input.at<cv::Vec3b>(x, yPrev).val[1]) / 2 +
        (input.at<cv::Vec3b>(x, yPrev).val[2] - input.at<cv::Vec3b>(x, yPrev).val[2]) / 2 ;

      // use absolute values because energy can be negative - uint "underflow"
      // see avidan et al., section 3 for reference
      int energy = abs(xEnergy) + abs(yEnergy);
      output.at<uchar>(x, y) = energy;
    }
  }
  return output;
}

cv::Mat calcCost(cv::Mat energy, int dir) {
  // calculate path cost matrix
  cv::Mat cost = cv::Mat::zeros(energy.size(), CV_32F);
  if(dir == HORI) {
    energy.col(0).copyTo(cost.col(0));
    for (int x = 1; x < energy.size().width; x++) {
      for (int y = 0; y < energy.size().height; y++) {
        int above = y;
        int below = y;
        if (y > 0) above--;
        if (y < energy.size().height) below++;

        int pixelcost = std::min(std::min(cost.at<uchar>(x - 1, above), cost.at<uchar>(x - 1, y)),
                            cost.at<uchar>(x - 1, below));
        cost.at<uchar>(x,y) = pixelcost + energy.at<uchar>(x, y);
      }
    }
  } else {
    energy.row(0).copyTo(cost.row(0));
  }
  cv::Mat cost8bit;
  cost.convertTo(cost8bit, CV_8UC1, 1);
  cv::imwrite("cost.jpg", cost8bit);
  return cost;
}
