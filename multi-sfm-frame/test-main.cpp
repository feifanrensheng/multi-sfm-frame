#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

#include <Eigen/Geometry> 
#include <boost/format.hpp>  // for formating strings
#include <pcl/point_types.h> 
#include <pcl/io/pcd_io.h> 
#include <pcl/visualization/pcl_visualizer.h>
//#include <tinydir.h>

#include "ARDrawingContext.hpp"
#include "GeometryTypes.hpp"


#include <GL/gl.h>
#include <GL/glu.h>



using namespace cv;
using namespace std;

void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector<vector<Vec3b>>& colors_for_all
	)
{
	key_points_for_all.clear();
	descriptor_for_all.clear();
	Mat image;

	//??ȡͼ??????ͼ????㣬??????
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);
	for (auto it = image_names.begin(); it != image_names.end(); ++it)
	{
		image = imread(*it);
		if (image.empty()) continue;

		cout << "Extracing features: " << *it << endl;

		vector<KeyPoint> key_points;
		Mat descriptor;
		//ż?????????ʧ?ܵĴ??
		sift->detectAndCompute(image, noArray(), key_points, descriptor);

		//???????????????
		if (key_points.size() <= 10) continue;

		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);

		vector<Vec3b> colors(key_points.size());
		for (int i = 0; i < key_points.size(); ++i)
		{
			Point2f& p = key_points[i].pt;
			colors[i] = image.at<Vec3b>(p.y, p.x);
		}
		colors_for_all.push_back(colors);
	}
}

void match_features(Mat& query, Mat& train, vector<DMatch>& matches)
{
	vector<vector<DMatch>> knn_matches;
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(query, train, knn_matches, 2);

	//?????Ratio Test????ƥ??ľ??
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		//Ratio Test
		if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
			continue;

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) min_dist = dist;
	}

	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		//???????Ratio Test?ĵ?ƥ????????
		if (
			knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			)
			continue;

		//????????
		matches.push_back(knn_matches[r][0]);
	}
}

void match_features_all(vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all)
{
	matches_for_all.clear();
	// n??ͼ???????˳???n-1 ????
	// 1?2ƥ???2?3ƥ???3?4ƥ????????
	for (int i = 0; i < descriptor_for_all.size() - 1; ++i)
	{
		cout << "Matching images " << i << " - " << i + 1 << endl;
		vector<DMatch> matches;
		match_features(descriptor_for_all[i], descriptor_for_all[i + 1], matches);
		matches_for_all.push_back(matches);
	}
}

bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)
{
	//????ڲξ??ȡ????Ľ????????꣨?????꣩
	double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	//??????????????????RANSAC????һ?????ʧ???
	Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
	if (E.empty()) return false;

	double feasible_count = countNonZero(mask);
	cout << (int)feasible_count << " -in- " << p1.size() << endl;
	//???ANSAC?????outlier??????0%ʱ????????ɿ???
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
		return false;

	//?ֽⱾ???󣬻????Ա任
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	//ͬʱλ????????ǰ???ĵ????Ҫ?????
	if (((double)pass_count) / feasible_count < 0.7)
		return false;

	return true;
}

void get_matched_points(
	vector<KeyPoint>& p1, 
	vector<KeyPoint>& p2, 
	vector<DMatch> matches, 
	vector<Point2f>& out_p1, 
	vector<Point2f>& out_p2
	)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		out_p2.push_back(p2[matches[i].trainIdx].pt);
	}
}

void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
	)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}

void get_matched_descriptors(Mat& descriptor, vector<DMatch>& matches, vector<vector<float>>& out_descriptor)
{
	out_descriptor.clear();
	vector<float> dest;
	for (int i = 0; i < matches.size(); i++)
	{
		for (int j = 0; j < descriptor.cols; j++)
		{
			dest.push_back(descriptor.at<float>(matches[i].queryIdx, j));
		}
		out_descriptor.push_back(dest);
	}
}

void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure)
{
	//??????????Ӱ???R T]??triangulatePointsֻ֧??loat?
	Mat proj1(3, 4, CV_32FC1);
	Mat proj2(3, 4, CV_32FC1);

	R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);
	T1.convertTo(proj1.col(3), CV_32FC1);

	R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	T2.convertTo(proj2.col(3), CV_32FC1);

	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK*proj1;
	proj2 = fK*proj2;

	//???ؽ?
	Mat s;
	triangulatePoints(proj1, proj2, p1, p2, s);

	structure.clear();
	structure.reserve(s.cols);
	for (int i = 0; i < s.cols; ++i)
	{
		Mat_<float> col = s.col(i);
		col /= col(3);	//?????꣬?Ҫ?????????Ԫ????????????
		structure.push_back(Point3f(col(0), col(1), col(2)));
	}
}

void maskout_points(vector<Point2f>& p1, Mat& mask)
{
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
	vector<Vec3b> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void maskout_descriptors(vector<vector<float>>& descriptors, Mat& mask)
{
	vector<vector<float>> descriptors_copy = descriptors;
	descriptors.clear();
	for (int i = 0; i < mask.rows; i++ )
	{
		if (mask.at<uchar>(i) > 0)
		descriptors.push_back(descriptors_copy[i]);
	}
}

void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, vector<Point3f>& structure, vector<Vec3b>& colors)
{
	int n = (int)rotations.size();

	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << (int)structure.size();
	
	fs << "Rotations" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << rotations[i];
	}
	fs << "]";

	fs << "Motions" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << motions[i];
	}
	fs << "]";

	fs << "Points" << "[";
	for (size_t i = 0; i < structure.size(); ++i)
	{
		fs << structure[i];
	}
	fs << "]";

	fs << "Colors" << "[";
	for (size_t i = 0; i < colors.size(); ++i)
	{
		fs << colors[i];
	}
	fs << "]";

	fs.release();
}

void get_objpoints_and_imgpoints(
	vector<DMatch>& matches,
	vector<int>& struct_indices, 
	vector<Point3f>& structure, 
	vector<KeyPoint>& key_points,
	vector<Point3f>& object_points,
	vector<Point2f>& image_points)
{
	object_points.clear();
	image_points.clear();

	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];
		if (struct_idx < 0) continue;

		object_points.push_back(structure[struct_idx]);
		image_points.push_back(key_points[train_idx].pt);
	}
}

void fusion_structure(
	vector<DMatch>& matches, 
	vector<int>& struct_indices, 
	vector<int>& next_struct_indices,
	vector<Point3f>& structure, 
	vector<Point3f>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors,
	vector<vector<float>>& descriptors,
	vector<vector<float>>& next_descriptors
	)
{
	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];
		if (struct_idx >= 0) //??õ?ڿռ??Ѿ???ڣ?????????Ӧ?Ŀռ?Ӧ????һ??????Ҫ?ͬ
		{
			next_struct_indices[train_idx] = struct_idx;
			continue;
		}

		//??õ?ڿռ??Ѿ???ڣ????õ?????ṹ????????????ռ?????Ϊ????????
		structure.push_back(next_structure[i]);
		colors.push_back(next_colors[i]);
		descriptors.push_back(next_descriptors[i]);
		struct_indices[query_idx] = next_struct_indices[train_idx] = structure.size() - 1;
	}
}

void init_structure(
	Mat K,
	vector<vector<KeyPoint>>& key_points_for_all, 
	vector<vector<Vec3b>>& colors_for_all,
	vector<vector<DMatch>>& matches_for_all,
	vector<Mat>& descriptor_for_all,
	vector<Point3f>& structure,
	vector<vector<int>>& correspond_struct_idx,
	vector<Vec3b>& colors,
	vector<vector<float>>& descriptors,
	vector<Mat>& rotations,
	vector<Mat>& motions
	)
{
	//????????ͼ?֮???任???
	vector<Point2f> p1, p2;
	vector<Vec3b> c2;
	Mat R, T;	//?ת???ƽ????
	Mat mask;	//mask????????????㣬??????????
	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);
	get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);
	get_matched_descriptors(descriptor_for_all[0], matches_for_all[0], descriptors);
	find_transform(K, p1, p2, R, T, mask);

	//对头两幅图像进行三维重建
	maskout_points(p1, mask);
	maskout_points(p2, mask);
	maskout_colors(colors, mask);
	maskout_descriptors(descriptors, mask);

	Mat R0 = Mat::eye(3, 3, CV_64FC1);
	Mat T0 = Mat::zeros(3, 1, CV_64FC1);
	reconstruct(K, R0, T0, R, T, p1, p2, structure);
	//?????????
	rotations = { R0, R };
	motions = { T0, T };

	//??correspond_struct_idx?Ĵ????ʼ??Ϊ?key_points_for_all?ȫһ?
	correspond_struct_idx.clear();
	correspond_struct_idx.resize(key_points_for_all.size());
	for (int i = 0; i < key_points_for_all.size(); ++i)
	{
		correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);
	}
	
	//?дͷ????ͼ??Ľṹ??
	int idx = 0;
	vector<DMatch>& matches = matches_for_all[0];
	for (int i = 0; i < matches.size(); ++i)
	{
		if (mask.at<uchar>(i) == 0)
			continue;

		correspond_struct_idx[0][matches[i].queryIdx] = idx;
		correspond_struct_idx[1][matches[i].trainIdx] = idx;
		++idx;
	}
}

void match_current_frame( Mat frame, Mat& train_descriptors, vector<Point3f>& structure, Mat& K, Mat& R, Mat& T)
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


	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		if (knn_matches[r][0].distance > 0.8*knn_matches[r][1].distance ) continue;

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
	//Mat r;
	solvePnPRansac(object_points, image_points, K, noArray(), R, T);

	
}

bool processFrame(const cv::Mat& cameraFrame, ARDrawingContext& drawingCtx, Transformation pose3d);
void processVideo(CameraCalibration& calibration, cv::VideoCapture& capture, Mat& train_descriptors, vector<Point3f>& structure);
void processSingleImage(CameraCalibration& calibration, Mat& train_descriptors, vector<Point3f>& structure, Mat image);

int main( int argc, char** argv)
{
	//vector<string> img_names;
	//get_file_names("images", img_names);
    vector<string> image_names = {"../0000.png", "../0001.png", "../0002.png", "../0003.png", "../0004.png", "../0005.png", "../0006.png", "../0007.png", "../0008.png", "../0009.png", "../0010.png"};
    //
    CameraCalibration calibration(2759.48f, 2764.16f, 1520.69, 1006.81);
    Mat K(Matx33d(
        2759.48, 0, 1520.69,
        0, 2764.16, 1006.81,
        0, 0, 1));

	vector<vector<KeyPoint>> key_points_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matches_for_all;
	//
	extract_features(image_names, key_points_for_all, descriptor_for_all, colors_for_all);
	//????ͼ????˳?ε???ƥ?
	match_features_all(descriptor_for_all, matches_for_all);

	vector<Point3f> structure;
	vector<vector<int>> correspond_struct_idx; //????i??ͼ??????????Ӧ??tructure?????
	vector<Vec3b> colors;
	vector<vector<float>> descriptors;
	vector<Mat> rotations;
	vector<Mat> motions;

	//??ʼ???ṹ???ά??ƣ?
	init_structure(
		K,
		key_points_for_all,
		colors_for_all,
		matches_for_all,
		descriptor_for_all,
		structure,
		correspond_struct_idx,
		colors,
		descriptors,
		rotations,
		motions
		);

	//?????ʽ???ʣ?????
	for (int i = 1; i < matches_for_all.size(); ++i)
	{
		vector<Point3f> object_points;
		vector<Point2f> image_points;
		Mat r, R, T;
		//Mat mask;

		//???????ͼ??ƥ???Ӧ???ά?㣬??????+1??ͼ????????ص?
		get_objpoints_and_imgpoints(
			matches_for_all[i], 
			correspond_struct_idx[i], 
			structure,
			key_points_for_all[i+1], 
			object_points,
			image_points
			);

		//????????
		solvePnPRansac(object_points, image_points, K, noArray(), r, T);
		//???ת???ת??Ϊ?ת???
		Rodrigues(r, R);
		//?????????
		rotations.push_back(R);
		motions.push_back(T);

		vector<Point2f> p1, p2;
		vector<Vec3b> c1, c2;
		vector<vector<float>> dests1;
		get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i], p1, p2);
		get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i], c1, c2);
		get_matched_descriptors(descriptor_for_all[i], matches_for_all[i], dests1);

		//根据之前求得的R，T进行三维重建
		vector<Point3f> next_structure;
		reconstruct(K, rotations[i], motions[i], R, T, p1, p2, next_structure);

		//将新的重建结果与之前的融合
		fusion_structure(
			matches_for_all[i], 
			correspond_struct_idx[i], 
			correspond_struct_idx[i + 1],
			structure, 
			next_structure,
			colors,
			c1,
			descriptors,
			dests1
			);
	}

	//????
	save_structure("structure.yml", rotations, motions, structure, colors);

	Mat train_descriptors = Mat(structure.size(), descriptor_for_all[0].cols, descriptor_for_all[0].type());
	for (int i =0; i < train_descriptors.rows; i++ )
	{
		for (int j = 0; j < train_descriptors.cols; j++ )
			train_descriptors.at<float>(i,j) = descriptors[i][j];
	}

	//
	Mat frame = imread(argv[1]);

	processSingleImage(calibration, train_descriptors, structure, frame );



    return 0;
}

void processSingleImage(CameraCalibration& calibration, Mat& train_descriptors, vector<Point3f>& structure, Mat image)
{
	cv::Size frameSize(image.cols, image.rows);
	//ARPipeline pipeline(patternImage, calibration);
	ARDrawingContext drawingCtx("Markerless AR", frameSize, calibration);
    Mat K(Matx33d(
        2759.48, 0, 1520.69,
        0, 2764.16, 1006.81,
        0, 0, 1));
	
	struct Transformation pose;
	struct Matrix33 pose_r;
	struct Vector3 pose_t;
	Mat R;
	Mat T;
	cv::Mat Rvec;
	cv::Mat_<float> Tvec;
	cv::Mat_<float> rotMat(3,3);
	Transformation pose3d;

	match_current_frame(image, train_descriptors, structure, K, R, T);
	R.convertTo(Rvec,CV_32F);
	T.convertTo(Tvec,CV_32F);
	cv::Rodrigues(Rvec, rotMat);

	// Copy to transformation matrix
	for (int col=0; col<3; col++)
	{
		for (int row=0; row<3; row++)
		{
			pose3d.r().mat[row][col] = rotMat(row,col); // Copy rotation component
		}
		pose3d.t().data[col] = Tvec(col); // Copy translation component
		pose3d = pose3d.getInverted();
	}
	
	bool shouldQuit = false;
	do 
	{
		shouldQuit = processFrame(image, drawingCtx, pose3d);
	} while (!shouldQuit);
}

void processVideo(CameraCalibration& calibration, cv::VideoCapture& capture, Mat& train_descriptors, vector<Point3f>& structure)
{
	// Grab first frame to get the frame dimensions
	cv::Mat currentFrame;
	capture >> currentFrame;

	// Check the capture succeeded:
	if (currentFrame.empty())
	{
		std::cout << "Cannot open video capture device" << std::endl;
		return;
	}

	cv::Size frameSize(currentFrame.cols, currentFrame.rows);
	ARDrawingContext drawingCtx("Markerless AR", frameSize, calibration);

	Mat R, T;
    Mat K(Matx33d(
        2759.48, 0, 1520.69,
        0, 2764.16, 1006.81,
        0, 0, 1));
	bool shouldQuit = false;
	struct Transformation pose;
	struct Matrix33 pose_r;
	struct Vector3 pose_t;
	cv::Mat Rvec;
	cv::Mat_<float> Tvec;
	cv::Mat_<float> rotMat(3,3);
	Transformation pose3d;


	do
	{
		capture >> currentFrame;
		if (currentFrame.empty())
		{
			shouldQuit = true;
			continue;
		}
		match_current_frame(currentFrame, train_descriptors, structure, K, R, T);
		R.convertTo(Rvec,CV_32F);
		T.convertTo(Tvec,CV_32F);
		cv::Rodrigues(Rvec, rotMat);

		// Copy to transformation matrix
		for (int col=0; col<3; col++)
		{
			for (int row=0; row<3; row++)
			{
				pose3d.r().mat[row][col] = rotMat(row,col); // Copy rotation component
			}
			pose3d.t().data[col] = Tvec(col); // Copy translation component
			pose3d = pose3d.getInverted();
		}
		shouldQuit = processFrame(currentFrame, drawingCtx, pose3d);
	} while (!shouldQuit);
	
}

bool processFrame(const cv::Mat& cameraFrame, ARDrawingContext& drawingCtx, Transformation pose3d)
{
	// Clone image used for background (we will draw overlay on it)
	cv::Mat img = cameraFrame.clone();

	// Set a new camera frame:
	drawingCtx.updateBackground(img);

	// Find a pattern and update it's detection status:
    drawingCtx.isPatternPresent = true;

    // Update a pattern pose:
    drawingCtx.patternPose = pose3d;

	// Request redraw of the window:
	drawingCtx.updateWindow();

	// Read the keyboard input:
	int keyCode = cv::waitKey(5);

	bool shouldQuit = false;
	if (keyCode == 27 || keyCode == 'q')
	{
		shouldQuit = true;
	}

	return shouldQuit;
}


