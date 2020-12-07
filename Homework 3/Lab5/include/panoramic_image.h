#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

//class implementing methods to generate a panoramic image from a set of input images
class PanoramicImage {

// Methods
public:
	
	/*
		default constructor
	*/
	PanoramicImage();

	/*
		load the set of input images
		path: path of the directory to load images from 
		format: format of the images contained in the directory (lower case e.g.: "png" for PNG images)
		angle: angle in degrees (the field of view of the camera)
	*/
	void loadData(char* path, char* format, int angle);

	/*
		generate the panoramic image from input
	*/
	void doPanoramic();

	/*
		return result image
	*/
	cv::Mat getResult();

	/*
		show result image until key is pressed
	*/
	void showResult();
	

// Data
protected:
	
	//vector of input images 
	std::vector<cv::Mat> images;

	//result image
	cv::Mat result;

	//fov of the camera with which the pictures were taken
	int fov;
};
