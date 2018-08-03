#include <opencv2/opencv.hpp>
#include "GeometryTypes.hpp"


void current_frame(Mat& frame, Mat& train_descriptors, vector<Point3f>& structure, Mat& K, Mat& R, Mat& T, Transformation& patternPose)
{
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);
	vector<KeyPoint> key_points;
	Mat descriptor;


	sift->detectAndCompute(frame, noArray(), key_points, descriptor);
	vector<vector<DMatch>> knn_matches;
	vector<DMatch> matches;

	FlannBasedMatcher matcher;
	matcher.knnMatch(descriptor, train_descriptors, knn_matches, 2);

	//Ratio Test
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
			continue;

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) min_dist = dist;

	}
	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		if (
			knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance || 
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			)
			continue;

		//
		matches.push_back(knn_matches[r][0]);
	}

	vector<Point3f> object_points;
	vector<Point2f> image_points;

	for (int i = 0; i < matches.size(); i++ )
	{
		object_points.push_back(structure[matches[i].trainIdx]);
		image_points.push_back(key_points[matches[i].queryIdx].pt);
	}

	cout<<object_points.size()<<endl;
	cout<<image_points.size()<<endl;

	//
	Mat r;
	solvePnPRansac(object_points, image_points, K, noArray(), r, T);
	Rodrigues(r, R);
	
	
}