#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void onCannyChange(int, void*);
void onHoughChangeL(int, void*);
void onHoughChangeC(int, void*);
void linesIntersection(vector<Vec2f> lines, Point &r);

//global variables
Mat input_img;
Mat img_gray;
Mat edges;
Mat lines_img;
Mat final_img;

Point points[4];
vector<Vec2f> linesPair;

int slider;
int lowThreshold;
int rho;
int thresh;
int maxRadius;



int main(int argc, char *argv[], char *envp[]) {
	
	//read and open input image
	input_img = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
	imshow("input image", input_img);
	waitKey(0);



	//generate the canny image (use trackbars to decide parameters to use)
	
	//first generate grayscale img
	cvtColor(input_img, img_gray, CV_BGR2GRAY);
	//initialise values for the trackbars
	lowThreshold = 10;
	slider = 1;
	edges = img_gray.clone();

	//tune parameters in order to highlight the right-most street lane's edges
	namedWindow("CANNY IMAGE");
	createTrackbar("low threshold", "CANNY IMAGE", &lowThreshold, 200, onCannyChange);
	createTrackbar("blur kernel size", "CANNY IMAGE", &slider, 10, onCannyChange);
	//start showing image
	onCannyChange(0, 0);

	waitKey(0);
	destroyWindow("CANNY IMAGE");



	//use HOUGH TRANSFORM to find the STRAIGHT lines
	namedWindow("STRAIGHT LINES");
	thresh = 50;
	rho = 1;
	createTrackbar("rho", "STRAIGHT LINES", &rho, 3, onHoughChangeL);
	createTrackbar("threshold", "STRAIGHT LINES", &thresh, 200, onHoughChangeL);
	
	//start showing image
	onHoughChangeL(0, 0);

	waitKey(0);

	//find the intersection between the two lines
	Point intersection;
	linesIntersection(linesPair, intersection);

	//save vertices of the triangle
	vector<Point> vertices;
	vertices.push_back(intersection);
	vertices.push_back(points[0]);
	vertices.push_back(points[3]);
	
	vector<vector<Point> > triangle; //this is only used to be passed to the fillPoly function
	triangle.push_back(vertices);

	//keep input image untouched, work on a copy
	input_img.copyTo(lines_img);
	//color the triangle
	fillPoly(lines_img, triangle, Scalar(0, 0, 255));
	imshow("RESULT", lines_img);
	
	waitKey(0);
	destroyAllWindows();



	//tune parameters in order to highlight the edges of the roadsign
	
	namedWindow("CANNY IMAGE");
	createTrackbar("low threshold", "CANNY IMAGE", &lowThreshold, 100, onCannyChange);
	createTrackbar("blur kernel size", "CANNY IMAGE", &slider, 10, onCannyChange);
	//start showing image
	onCannyChange(0, 0);

	waitKey(0);
	


	//now use hough transform to FIND THE CIRCLE lines in the image of edges
	namedWindow("CIRCLES");
	thresh = 20;
	maxRadius = 10;
	createTrackbar("threshold", "CIRCLES", &thresh, 50, onHoughChangeC);
	createTrackbar("Max radius", "CIRCLES", &maxRadius, 100, onHoughChangeC);

	//start showing image
	onHoughChangeC(0, 0);

	waitKey(0);
	destroyAllWindows();



	//show final triangle and circle on the image
	imshow("FINAL RESULT", final_img);
	waitKey(0);
}



// trackbar callback function for canny image
void onCannyChange(int, void*) {
	int ratio = 3;
	int highThreshold = lowThreshold * ratio;
	//kernel size must be odd
	int kernel_size = 2 * slider + 1;

	// Reduce noise with a variable kernel
	blur(img_gray, edges, Size(kernel_size, kernel_size));
	
	// Canny detector
	Canny(edges, edges, lowThreshold, highThreshold, 3);
	
	imshow("CANNY IMAGE", edges);
}



//traackbar callback function used to find main STRAIGHT lines in the image
void onHoughChangeL(int, void*) {
	//lines found by hough transofrm will be stored here
	vector<Vec2f> lines;
	//keep input image untouched, work on a copy
	input_img.copyTo(lines_img);
	
	//find lines
	HoughLines(edges, lines, rho, CV_PI / 180, thresh);
	
	// Draw the two main lines
	for (size_t i = 0; i < 2; i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		//draw lines in the image
		line(lines_img, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}

	//update the first two lines that were found
	linesPair.swap(lines);
	
	imshow("STRAIGHT LINES", lines_img);
}



//trackbar callback function used to find CIRCLES in the image
void onHoughChangeC(int, void*) {
	vector<Vec3f> circles;
	
	//every time we restore final_img to lines_img because we don't want keep showing previously found circles
	lines_img.copyTo(final_img);	

	//find circles
	HoughCircles(edges, circles, CV_HOUGH_GRADIENT, 1, 1, lowThreshold*3, thresh, 0, maxRadius);

	//draw FIRST detected circle
	if (circles.size()>0) {
		size_t i = 0;
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle
		circle(final_img, center, radius, Scalar(0, 255, 0), -1, 8);
	}

	imshow("CIRCLES",final_img);
}



// support funciton used to find the intersection point of TWO lines
void linesIntersection(vector<Vec2f> lines, Point &r) {
	//need at least 2 lines in the vector
	if (lines.size() >= 2) {
		//convert lines to (x,y) coordinates
		for (size_t i = 0; i < 2; i++)
		{
			float rho = lines[i][0], theta = lines[i][1];

			double a = cos(theta), b = sin(theta);
			double x0 = a * rho, y0 = b * rho;
			points[2 * i].x = cvRound(x0 + 1000 * (-b));
			points[2 * i].y = cvRound(y0 + 1000 * (a));
			points[2 * i + 1].x = cvRound(x0 - 1000 * (-b));
			points[2 * i + 1].y = cvRound(y0 - 1000 * (a));
		}

		//The lines are defined by (o1, p1) and (o2, p2)
		//save lines points in global variable, we'll need point[0] and points[3] as vertices of the triangle inside the main
		Point2f o1 = points[0];
		Point2f o2 = points[2];
		Point2f p1 = points[1];
		Point2f p2 = points[3];


		//calculate intersection point coordinates
		Point2f x = o2 - o1;
		Point2f d1 = p1 - o1;
		Point2f d2 = p2 - o2;

		float cross = d1.x*d2.y - d1.y*d2.x;
		if (abs(cross) < /*EPS*/1e-8)
			return;

		double t1 = (x.x * d2.y - x.y * d2.x) / cross;
		r = o1 + d1 * t1;
	}
}