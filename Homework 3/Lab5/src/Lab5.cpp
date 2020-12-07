#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "panoramic_image.h"

using namespace cv;
using namespace std;

int main(int argc, char *argv[], char *envp[]) {
	
	int fov = atoi(argv[3]);

	PanoramicImage panorama = PanoramicImage();
	
	panorama.loadData(argv[1], argv[2], fov);
	panorama.doPanoramic();
	panorama.showResult();

}
