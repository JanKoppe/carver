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

  std::cout << "Calculating energy function for image" << std::endl;
  cv::Mat energy = calcEnergy(image);

  std:: cout << "Write image of energy function to energy.jpg" << std::endl;
  cv::imwrite("energy.jpg", energy);

  return 0;
}

cv::Mat calcEnergy(cv::Mat input) {
  cv::Mat output = cv::Mat(input.size(), input.type());
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

      int xEnergy =
        (input.at<cv::Vec3b>(xPrev, y).val[0] - input.at<cv::Vec3b>(xNext, y).val[0]) / 2 +
        (input.at<cv::Vec3b>(xPrev, y).val[1] - input.at<cv::Vec3b>(xNext, y).val[1]) / 2 +
        (input.at<cv::Vec3b>(xPrev, y).val[2] - input.at<cv::Vec3b>(xNext, y).val[2]) / 2 ;
      int yEnergy =
        (input.at<cv::Vec3b>(x, yPrev).val[0] - input.at<cv::Vec3b>(x, yPrev).val[0]) / 2 +
        (input.at<cv::Vec3b>(x, yPrev).val[1] - input.at<cv::Vec3b>(x, yPrev).val[1]) / 2 +
        (input.at<cv::Vec3b>(x, yPrev).val[2] - input.at<cv::Vec3b>(x, yPrev).val[2]) / 2 ;

      int energy = abs(xEnergy) + abs(yEnergy);

      output.at<cv::Vec3b>(x, y) = cv::Vec3b(energy, energy, energy);
    }
  }
  return output;
}
