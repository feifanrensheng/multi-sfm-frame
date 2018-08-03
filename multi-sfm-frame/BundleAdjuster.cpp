#include "BundleAdjuster.hpp"
#include "Common.hpp"

#include <Math/v3d_linear.h>
#include <Base/v3d_vrmlio.h>
#include <Geometry/v3d_metricbundle.h>

using namespace V3D;
using namespace std;
using namespace cv;

void BundleAdjuster(vector<CloudPoint>& pointcloud,
					Mat& cam_matrix,
					const std::vector<std::vector<cv::KeyPoint>>& imgpts,
					std::map<int, cv::Matx34d>& Pmats
					)
{
	int N = Pmats.size(), M = pointcloud.size();

	// convert camera intrinsics to BA datastructs
	Matrix3x3d KMat;
	make
}