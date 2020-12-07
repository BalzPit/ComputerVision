#include "panoramic_image.h"
#include "panoramic_utils.h"

using namespace cv;
using namespace std;

//constructor
PanoramicImage::PanoramicImage() {}



void PanoramicImage::loadData(char* path, char* format, int angle) {
	//save fov 
	fov = angle;
	
	//file names
	vector<cv::String> fn;

	//create string of path with \*.format suffix
	stringstream stringStream;
	stringStream << path << "\\*." << format;
	
	string pth = stringStream.str();

	//save all names in the path ending with .png or .bmp
	glob(pth, fn, false);

	size_t count = fn.size(); //number of png files in images folder
	//load all png images from names
	for (size_t i = 0; i < count; i++) {
		Mat src = imread(fn[i]);
		images.push_back(src);
	}
}




void PanoramicImage::doPanoramic() {
	//vector of cylindrical projections
	vector<Mat> projections;

	//exectue cylinder projection of the images
	size_t count = images.size();
	for (size_t i = 0; i < count; i++) {
		Mat img = images.at(i);

		//save planar image of projection of input image on cylinder surface
		PanoramicUtils a = PanoramicUtils();
		Mat proj = a.cylindricalProj(img, fov / 2);

		projections.push_back(proj);
	}

	Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();

	//vectors that will hold descriptors and keypoints for each image
	vector<vector<KeyPoint>> keypoints;
	vector<Mat> descriptors;
	
	//compute keypoints and descriptors for each image
	for (size_t i = 0; i < count; i++) {
		Mat img = projections.at(i);
		vector<KeyPoint> keypoint;
		Mat descriptor;
		
		detector->detect(img, keypoint);
		detector->compute(img, keypoint, descriptor);
	
		//insert into vectors
		keypoints.push_back(keypoint);
		descriptors.push_back(descriptor);
	}
	
	// create Brute Force Matcher 
	BFMatcher matcher(NORM_L2);

	vector<vector<DMatch>> matches;	//matches.at(i) contains all matches between picture i and i+1
	vector<vector<DMatch>> refinedMatches;
	int rate = 9;	//refinement threshold rate

	vector<int> Xtranslation;
	vector<int> Ytranslation;
	
	//match consecutive images
	for (size_t i = 0; i < count-1; i++) {
		//take descriptors of 2 consecutive images
		Mat descriptor1 = descriptors.at(i);
		Mat descriptor2 = descriptors.at(i + 1);
		
		//find matches between descriptors
		vector<DMatch> match;
		matcher.match(descriptor1, descriptor2, match);

		//save matches of 2 consecutive pictures
		matches.push_back(match);
		printf("matches: %i\n", match.size());
		
		//find min dist
		int mindist = 100;
		for (size_t j = 0; j < matches.at(i).size(); j++) {
			int dist = matches.at(i).at(j).distance;
			if (dist < mindist) {
				mindist = dist;	
			}
		}
		printf("mindist: %i\n", mindist);
		
		//refine matches
		vector<DMatch> refined;
		for (size_t j = 0; j < matches.at(i).size(); j++) {
			//if there is a not so strong match (high distance) don't count it
			if (matches.at(i).at(j).distance < mindist*rate) {
				refined.push_back(matches.at(i).at(j));
			}
		}
		//save refined vector of matches between consecutive images
		refinedMatches.push_back(refined);
		printf("refined matches size: %i\n", refined.size());


		//FIND TRANSLATIONS between consecutive images

		vector<Point2f> img1;
		vector<Point2f> img2;

		for (size_t j = 0; j < refinedMatches.at(i).size(); j++)
		{
			//Get the keypoints from the good matches
			img1.push_back(keypoints.at(i)[refinedMatches.at(i)[j].queryIdx].pt);
			img2.push_back(keypoints.at(i+1)[refinedMatches.at(i)[j].trainIdx].pt);
		}
		Mat inliers; //save inliers of consecutive images here (binary matrix of 1 column and n=refinedMatches.at(i).size() rows) 
		Mat homography = findHomography(img1, img2, inliers, CV_RANSAC);

		//get translations between inliers
		int xTrans = 0;
		int yTrans = 0;
		int tot = 0;

		for (size_t j = 0; j < inliers.rows; j++) {
			if (inliers.at<int>(j, 0) == 1) {
				// match j is an inlier if it has value 1
				Point2f p1 = keypoints.at(i)[refinedMatches.at(i)[j].queryIdx].pt;
				Point2f p2 = keypoints.at(i + 1)[refinedMatches.at(i)[j].trainIdx].pt;
				
				//get x and y coordinates difference between the 2 points
				xTrans += (p1.x - p2.x);
				yTrans += (p1.y - p2.y);
				tot++;
			}
		}

		if (tot != 0) {
			xTrans = xTrans / tot;
			yTrans = yTrans / tot;
		}

		printf("average X translation for image %i and %i: %i\n", i, i+1, xTrans);
		printf("average Y translation for image %i and %i: %i\n", i, i+1, yTrans);

		//save average of translations between image i and i+1
		Xtranslation.push_back(xTrans);
		Ytranslation.push_back(yTrans);
	}

	
	//prepare matrix
	//calculate sums for columns keeping in mind the translations differences
	int finalcols = 0;
	int finalrows = projections.at(0).rows * 2;

	for (size_t i = 0; i < count-1; i++) {
		//finalcols += projections.at(i).cols + Xtranslation.at(i);
		finalcols += Xtranslation.at(i);
	}
	// add columns of the final image in the vector
	finalcols += projections.at(count-1).cols;
	printf("final columns: %i\n", finalcols);

	//matrix of final image
	Mat final_img = Mat(finalrows, finalcols, projections.at(0).type());

	//range boundaries for the submatrix
	int left = 0;
	int right = 0;
	int top = finalrows/4; //center images in final matrix
	int bottom;

	//limits to show the submatrix of final matrix that doesn't contain blank parts
	int topLimit = 0;
	int bottomLimit = finalrows;

	Mat sub;
	Mat nextImg;
	
	//MERGE INPUT IMAGES TO COMPUTE FINAL PANORAMA
	for (size_t i = 0; i < count-1; i++) {
		nextImg = projections.at(i);

		//equalize image before copying it
		equalizeHist(nextImg, nextImg);

		bottom = top + nextImg.rows;
		right = left+ nextImg.cols;
		//selelct a submatrix of the final image
		sub = final_img(Range(top, bottom), Range(left, right));

		//copy image into submatrix
		nextImg.copyTo(sub);

		//set boundaries of next image keeping in mind the translation
		top = top + Ytranslation.at(i);
		left = left + Xtranslation.at(i);

		//update limits
		if (top > topLimit) 
			topLimit = top;
		if (bottom < bottomLimit) 
			bottomLimit = bottom;
	}
	//add last image
	nextImg = projections.at(count-1);
	bottom = top + nextImg.rows;
	right = left + nextImg.cols;
	sub = final_img(Range(top, bottom), Range(left, right));
	nextImg.copyTo(sub);

	//take the submatrix without blank parts
	sub = final_img(Range(topLimit, bottomLimit), Range(0, finalcols));
	//save final result
	sub.copyTo(result);
}



Mat PanoramicImage::getResult() {
	return result;
}



void PanoramicImage::showResult() {
	namedWindow("PANORAMIC IMAGE", WINDOW_NORMAL);
	imshow("PANORAMIC IMAGE", result);
	waitKey(0);
}
