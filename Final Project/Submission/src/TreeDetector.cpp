#include "tree_detector.h"



//empty constructor
TreeDetector::TreeDetector() {}



//constructor
TreeDetector::TreeDetector(char* path) {
	TreeDetector::setCascade(path);
}



//set cascade for detector
int TreeDetector::setCascade(char* path) {
	
	//Load cascade
	int n = tree_cascade.load(path);

	if (!n){
		return -1;
	}

	return 1;
}



//perform the detection
void TreeDetector::doDetection(Mat img) {

	namedWindow("TREES detection", CV_WINDOW_NORMAL);
	imshow("TREES detection", img);
	waitKey();
	
	Mat img_gray, img_blur;
	int x = img.cols;
	int y = img.rows;
	int blurSize = x / 60;

	//image Pre-Processing

	cvtColor(img, img_gray, COLOR_BGR2GRAY);
	blur(img_gray, img_blur, Size(blurSize, blurSize));
	//bilateralFilter(img_gray, img_blur, 20, x/9, x/9);
	equalizeHist(img_blur, img_blur);

	Mat hsv_img, result;

	//thresholds for H S and V range
	Scalar low = Scalar(80, 0, 0);
	Scalar high = Scalar(255, 255, 255);

	//convert to hsv color space
	cvtColor(img, hsv_img, COLOR_BGR2HSV);
	//select a range of colors to mask
	inRange(hsv_img, low, high, result);

	//apply mask on the image
	bitwise_or(img_blur, result, img_blur);

	//threshold on minimum window size
	int	sizeXThreshold = x / 8;
	int	sizeYThreshold = y / 8;

	//minimum window im which to look for tree
	Size minSize = Size(sizeXThreshold, sizeYThreshold);
	Size maxSize = Size(x, y);

	vector<Rect> trees;
	
	// Detect trees
	
	printf("trees detection...");
	tree_cascade.detectMultiScale(img_blur, trees, 1.1, 10, 0, minSize);
	printf("   Completed\n");
	printf("number of trees in the image= %i\n", trees.size());

	Scalar color = Scalar(0, 150, 255);
	for (size_t i = 0; i < trees.size(); i++)
	{
		//draw bounding box on the tree
		Point topLeft(trees[i].x, trees[i].y);
		Point topRight(trees[i].x + trees[i].width, trees[i].y);
		Point bottomLeft(trees[i].x, trees[i].y + trees[i].height);
		Point bottomRight(trees[i].x + trees[i].width, trees[i].y + trees[i].height);

		line(img, topLeft, bottomLeft, color, 3);
		line(img, topLeft, topRight, color, 3);
		line(img, topRight, bottomRight, color, 3);
		line(img, bottomLeft, bottomRight, color, 3);
	}

	//draw a red cross accros the screen when no images are found
	if (trees.size() == 0) {
		line(img, Point(0, 0), Point(img.cols, img.rows), Scalar(0, 0, 255), 3);
		line(img, Point(img.cols, 0), Point(0, img.rows), Scalar(0, 0, 255), 3);
	}

	//show result
	namedWindow("TREES detection", CV_WINDOW_NORMAL);
	imshow("TREES detection", img);
	waitKey(0);
}