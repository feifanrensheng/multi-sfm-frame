#include "pose_estimation_3d2d.hpp"

void bundleAddjustment (
	const vector<Point3f> points_3d,
	const vector<Point2f> points_2d,
	const Mat& K,
	Mat& R, Mat& t 
)
{
	// 初始化g2o
	typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;   // pose 维度为 6, landmark 维度为 3
	Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
	Block* solver_ptr = new Block( linearSolver ); // 矩阵块求解器
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm ( solver );

	// vertex
	g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Exmap(); // camera pose
	Eigen::Matrix3d R_mat;
	R_mat <<
			R.at<double> (0,0), R.at<double> (0,1), R.at<double> (0,2),
			R.at<double> (1,0), R.at<double> (1,1), R.at<double> (1,2),
			R.at<double> (2,0), R.at<double> (2,1), R.at<double> (2,2);
	pose->setId (0);
	pose->setEstimate (g2o::SE3Quat (
						R_mat,
						Eigen::Vector3d (t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0))
						));

	optimizer.addVertex(pose);

	int index = 1;
	for ( const Point3f p:points_3d ) // landmarks
	{
		g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
		point->setId(index++);
		point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z ));
		point->setMarginalized( true ); // g2o 中必须设置 marg
		optimizer.addVertex ( point );
	}

	// parameter: camera intrinsics
	g2o::CameraParameters* camera = new g2o::CameraParameters(
		K.at<double> (0,0), Eigen::Vector2d (K.at<double> (0,2), K.at<double>(1,2)), 0 );
	camera->setId(0);
	optimizer.addParameter(camera);

	// edges
	index = 1;
	for ( const Point2f p:points_2d )
	{
		g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
		edge->setId(index);
		edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index )));
		edge->setVertex( 1, pose );
		edge->setMeasurement ( Eigen::Vector2d (p.x, p.y ) );
		edge->setParameterId (0,0);
		edge->setInformation ( Eigen::Matrix2d::Identity() );
		optimizer.addEdge ( edge );
		index++;
	}

	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	optimizer.setVerbose ( true );
	optimizer.initializeOptimization();
	optimizer.optimize (100);
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
	cout<<"optimization costs time: "<<time_used.count() << " second."<<endl;

	cout<<endl<<"after optimization:"<<endl;
	cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() << endl;


}

