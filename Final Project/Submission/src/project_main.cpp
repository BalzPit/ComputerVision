#include <iostream>
#include <tree_detector.h>

using namespace cv;
using namespace std;


int main(int argc, char *argv[], char *envp[]) {

	//load input paths from command line arguments
	char * cascadePath = argv[1];
	String imgPath = argv[2];
	
	//file names
	vector<cv::String> fn;

	//create string of path with \*.format suffix
	stringstream stringStream;
	stringStream << imgPath << "\\*.jpg";
	string pth = stringStream.str();
	
	//save all names in the path ending with .jpg
	glob(pth, fn, false);
	size_t count = fn.size(); //number of jpg files in images folder

	//initialise tree detector obj
	TreeDetector tDetector = TreeDetector::TreeDetector();
	//set cascade to be used
	if (tDetector.setCascade(cascadePath) == -1)
	{
		cout << "--(!)Error loading trees cascade\n";
		return -1;
	};

	//Start tree detection on each image
	for (size_t i = 0; i < count; i++) {
		
		Mat image = imread(fn[i]);

		//perform detection
		tDetector.doDetection(image);
	}

	return 0;
}
