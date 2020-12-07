#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "filter.h"

using namespace cv;
using namespace std;

vector<Mat> calculateHistograms(Mat bgr[3]);
Mat pushandMerge(Mat channels[3], Mat merged);
void showHistogram(vector<Mat>& hists);
void onMedianChange(int, void*);
void onGaussianChange(int, void*);
void onBilateralChange(int, void*);

//structure used to pass parameters for filters
struct ImageWithParams
{
	int param1;
	int param2;
	int param3;
	cv::Mat src_img, filtered_img;
};



int main(int argc, char *argv[], char *envp[]) {

	Mat img;
	Mat resized_img;
	Mat bgr[3];
	Mat resized_bgr[3];
	
	img = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);

	//resize image to fit screen and show
	resize(img, resized_img, Size(), 0.25, 0.25);

	// split RGB values from img (opencv uses bgr order to store)
	split(img, bgr);

	//resize bgr as well
	resize(bgr[0], resized_bgr[0], Size(), 0.25, 0.25);
	resize(bgr[1], resized_bgr[1], Size(), 0.25, 0.25);
	resize(bgr[2], resized_bgr[2], Size(), 0.25, 0.25);

	imshow("image", resized_img);
	imshow("imageB", resized_bgr[0]);
	imshow("imageG", resized_bgr[1]);
	imshow("imageR", resized_bgr[2]);

	waitKey(0);

	//caluclate and show histograms of the bgr channels
	vector<Mat> channels_vector = calculateHistograms(bgr);
	showHistogram(channels_vector);
	
	waitKey(0);

	Mat equalized_img;
	Mat equalized_bgr[3];

	//equalize the three channels separately
	equalizeHist(bgr[0], equalized_bgr[0]);
	equalizeHist(bgr[1], equalized_bgr[1]);
	equalizeHist(bgr[2], equalized_bgr[2]);

	//merge channels and show equalized image
	equalized_img = pushandMerge(equalized_bgr, equalized_img);
	resize(equalized_img, equalized_img, Size(), 0.25, 0.25);
	imshow("Equalized image", equalized_img);

	//calculate and show equalized histograms
	channels_vector = calculateHistograms(equalized_bgr);
	showHistogram(channels_vector);
	
	waitKey(0);

	Mat hsv_img;
	Mat hsv[3];

	//equalize image using a different color space -> HSV
	cvtColor(img, hsv_img, COLOR_BGR2HSV);
	split(hsv_img, hsv);

	//equalize one channel (hsv[0] = Hue, hsv[1] = Saturation, hsv[2] = Value (Intensity))
	equalizeHist(hsv[2], hsv[2]);
	
	//merge the equalized channels, go back to rgb and show new image
	hsv_img = pushandMerge(hsv, hsv_img);
	cvtColor(hsv_img, hsv_img, COLOR_HSV2BGR);

	resize(hsv_img, hsv_img, Size(), 0.25, 0.25);
	imshow("HSV Image", hsv_img);

	waitKey(0);
	destroyAllWindows();



	//DENOISE THE EQUALIZED IMAGE BY APPLYING DIFFERENT FILTERS



	//	MEDIAN filter

	//create and fill struct
	ImageWithParams median;
	median.param1 = 0;	//kernel size
	median.src_img = hsv_img;
	
	//define window and trackbar assigned to it
	namedWindow("MEDIAN FILTER", CV_WINDOW_NORMAL);
	createTrackbar("kernel", "MEDIAN FILTER", &median.param1, 20, onMedianChange, &median);
	//start showing image
	onMedianChange(0, &median);
	
	waitKey(0);
	destroyWindow("MEDIAN FILTER");
	


	//GAUSSIAN BLUR filter

	//create and fill struct
	ImageWithParams gaussian;
	gaussian.param1 = 0;	//kernel size
	gaussian.param2 = 0;	//sigma
	gaussian.src_img = hsv_img;

	//define window and trackbar assigned to it
	namedWindow("GAUSSIAN FILTER");
	createTrackbar("kernel", "GAUSSIAN FILTER", &gaussian.param1, 20, onGaussianChange, &gaussian);
	createTrackbar("sigma", "GAUSSIAN FILTER", &gaussian.param2, 20, onGaussianChange, &gaussian);
	//start showing image
	onGaussianChange(0, &gaussian);

	waitKey(0);
	destroyWindow("GAUSSIAN FILTER");
	


	//	BILATERAL filter

	//create and fill struct
	ImageWithParams bilateral;
	bilateral.param1 = 10;  // fixed kernel size 
	bilateral.param2 = 0;	// sigma range
	bilateral.param3 = 0;	// sigma space
	bilateral.src_img = hsv_img;
	
	namedWindow("BILATERAL FILTER");
	createTrackbar("s range", "BILATERAL FILTER", &bilateral.param2, 200, onBilateralChange, &bilateral);
	createTrackbar("s space", "BILATERAL FILTER", &bilateral.param3, 200, onBilateralChange, &bilateral);
	//start showing image
	onBilateralChange(0, &bilateral);

	waitKey(0);

	return 0;
}



//this function is used to calculate histograms for each channel of a bgr (or any 3-channel) image
vector<Mat> calculateHistograms(Mat bgr[3]) {

	//HISTOGRAM CALCULATION
	int histSize = 256;
	float range[] = { 0, 256 };		//the upper boundary is exclusive
	const float* histRange = { range };
	bool uniform = true, accumulate = false;

	Mat hist[3];

	//calculate histogram separately for each channel (original and not resized version)
	calcHist(&bgr[0], 1, 0, Mat(), hist[0], 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr[1], 1, 0, Mat(), hist[1], 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr[2], 1, 0, Mat(), hist[2], 1, &histSize, &histRange, uniform, accumulate);
	
	//vector with the histograms to be returned
	vector<Mat> hist_vect;
	hist_vect.push_back(hist[0]);
	hist_vect.push_back(hist[1]);
	hist_vect.push_back(hist[2]);

	return hist_vect;
}



// hists = vector of 3 cv::mat of size nbins=256 with the 3 histograms
// e.g.: hists[0] = cv:mat of size 256 with the red histogram
//       hists[1] = cv:mat of size 256 with the green histogram
//       hists[2] = cv:mat of size 256 with the blue histogram
void showHistogram(std::vector<cv::Mat>& hists)
{
  // Min/Max computation
  double hmax[3] = {0,0,0};
  double min;
  cv::minMaxLoc(hists[0], &min, &hmax[0]);
  cv::minMaxLoc(hists[1], &min, &hmax[1]);
  cv::minMaxLoc(hists[2], &min, &hmax[2]);

  std::string wname[3] = { "blue", "green", "red" };
  cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
                           cv::Scalar(0,0,255) };

  std::vector<cv::Mat> canvas(hists.size());

  // Display each histogram in a canvas
  for (int i = 0, end = hists.size(); i < end; i++)
  {
    canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

    for (int j = 0, rows = canvas[i].rows; j < hists[0].rows-1; j++)
    {
      cv::line(
            canvas[i],
            cv::Point(j, rows),
            cv::Point(j, rows - (hists[i].at<float>(j) * rows/hmax[i])),
            hists.size() == 1 ? cv::Scalar(200,200,200) : colors[i],
            1, 8, 0
            );
    }

    cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
  }
}



//function used to merge 3 chanel Mat into one
Mat pushandMerge(Mat channels[3], Mat merged) {

	//create a vector of Mat that is used to merge bgr channels
	vector<Mat> channels_vector;
	channels_vector.push_back(channels[0]);
	channels_vector.push_back(channels[1]);
	channels_vector.push_back(channels[2]);

	merge(channels_vector, merged);

	return merged;
}



void onMedianChange(int i, void* v) {
	ImageWithParams* median = (ImageWithParams*) v;

	// kernel size needs to be odd
	int kernel_size = 2 * median->param1 + 1;

	//initialise Median Filter obj
	MedianFilter median_filter = MedianFilter(median->src_img, kernel_size);

	//apply filter
	median_filter.doFilter();

	//show filtering result
	imshow("MEDIAN FILTER", median_filter.getResult());

}



void onGaussianChange(int i, void* v) {
	ImageWithParams* gaussian = (ImageWithParams*)v;

	// kernel size needs to be odd
	int kernel_size = 2 * gaussian->param1 + 1;

	//initialise Gaussian Filter obj
	GaussianFilter gaussian_filter = GaussianFilter(gaussian->src_img, kernel_size, gaussian->param2);

	//apply filter and show result
	gaussian_filter.doFilter();
	imshow("GAUSSIAN FILTER", gaussian_filter.getResult());
}



void onBilateralChange(int i, void* v) {
	ImageWithParams* bilateral = (ImageWithParams*)v;

	//initialise Gaussian Filter obj
	BilateralFilter bilateral_filter = BilateralFilter(bilateral->src_img, bilateral->param1, bilateral->param2, bilateral->param3);

	//apply filter and show result
	bilateral_filter.doFilter();
	imshow("BILATERAL FILTER", bilateral_filter.getResult());
}