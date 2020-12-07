#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

using namespace std;
using namespace cv;

//class implementing methods to generate a panoramic image from a set of input images
class TreeDetector {

	//Public Methods
public:

	/*
		default empty constructor
	*/
	TreeDetector();

	/*
		constructor
	*/
	TreeDetector(char* path);

	/*
		set the cascade outside of constructor
	*/
	int setCascade(char* path);

	/*
		perform detection on a given image
	*/
	void doDetection(Mat img);

	//Data
protected:

	CascadeClassifier tree_cascade;

};
