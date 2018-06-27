#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
Mat cameraMatrixL = Mat_<double>(3, 3);
Mat distCoeffL = Mat_<double>(5, 1);
/*
事先标定好的右相机的内参矩阵
fx 0 cx
0 fy cy
0 0  1
*/
Mat cameraMatrixR = Mat_<double>(3, 3);
Mat distCoeffR = Mat_<double>(5, 1);
int bdleft()
{
	ifstream fin("calibdata_left.txt"); /* 标定所用图像文件的路径 */
										//读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化	
	cout << "Start to extract corner points………………" << endl;
	int image_count = 0;  /* 图像数量 */
	Size image_size;  /* 图像的尺寸 */
	Size board_size = Size(11, 8);    /* 标定板上每行、列的角点数 */
	vector<Point2f> image_points_buf;  /* 缓存每幅图像上检测到的角点 */
	vector<vector<Point2f> > image_points_seq; /* 保存检测到的所有角点 */
	string filename;
	int count = -1;//用于存储角点个数。
	while (getline(fin, filename))
	{
		image_count++;
		// 用于观察检验输出
		cout << "image_count = " << image_count << endl;
		/* 输出检验*/
		//cout << "-->count = " << count;
		Mat imageInput = imread(filename);
		if (image_count == 1)  //读入第一张图片时获取图像宽高信息
		{
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;

		}

		/* 提取角点 */
		if (0 == findChessboardCorners(imageInput, board_size, image_points_buf))
		{
			cout << "can not find chessboard corners!\n"; //找不到角点
			exit(1);
		}
		else
		{
			Mat view_gray;
			cvtColor(imageInput, view_gray, CV_RGB2GRAY);
			/* 亚像素精确化 */
			find4QuadCornerSubpix(view_gray, image_points_buf, Size(5, 5)); //对粗提取的角点进行精确化
																			//cout << image_points_buf << endl;
			image_points_seq.push_back(image_points_buf);  //保存亚像素角点	
		}
	}
	int total = image_points_seq.size();
	cout << "total = " << total << endl;
	int CornerNum = board_size.width*board_size.height;  //每张图片上总的角点
	cout << CornerNum << endl;
	cout << "image_size.width = " << image_size.width << endl;
	cout << "image_size.height = " << image_size.height << endl;
	cout << "Corner extraction completed" << endl;

	//以下是摄像机标定
	cout << "Start calibration………………" << endl;
	/*棋盘三维信息*/
	Size square_size = Size(20, 20);  /* 实际测量得到的标定板上每个棋盘格的大小 */
	vector<vector<Point3f> > object_points; /* 保存标定板上角点的三维坐标 */
										   /*内外参数*/
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 摄像机内参数矩阵 */
															/*
															事先标定好的右相机的内参矩阵
															fx 0 cx
															0 fy cy
															0 0  1
															*/
	vector<int> point_counts;  // 每幅图像中角点的数量
	Mat distCoeffs = Mat(5, 1, CV_32FC1, Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
	vector<Mat> tvecsMat;  /* 每幅图像的旋转向量 */
	vector<Mat> rvecsMat; /* 每幅图像的平移向量 */
						  /* 初始化标定板上角点的三维坐标 */
	int i, j, t;
	for (t = 0; t<image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i<board_size.height; i++)
		{
			for (j = 0; j<board_size.width; j++)
			{
				Point3f realPoint;
				/* 假设标定板放在世界坐标系中z=0的平面上 */
				realPoint.x = i * square_size.width;
				realPoint.y = j * square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	/* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
	for (i = 0; i<image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}
	/* 开始标定 */
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
	cout << "Calibration completed\n";
	//对标定结果进行评价
	cameraMatrixL = cameraMatrix;
	distCoeffL = distCoeffs;

	return 0;
}
int bdright()
{
	ifstream fin("calibdata_right.txt"); /* 标定所用图像文件的路径 */
	cout << "Start to extract corner points………………" << endl;
	int image_count = 0;  /* 图像数量 */
	Size image_size;  /* 图像的尺寸 */
	Size board_size = Size(11, 8);    /* 标定板上每行、列的角点数 */
	vector<Point2f> image_points_buf;  /* 缓存每幅图像上检测到的角点 */
	vector<vector<Point2f> > image_points_seq; /* 保存检测到的所有角点 */
	string filename;
	int count = -1;//用于存储角点个数。
	while (getline(fin, filename))
	{
		image_count++;
		// 用于观察检验输出
		cout << "image_count = " << image_count << endl;
		/* 输出检验*/
		//cout << "-->count = " << count;
		Mat imageInput = imread(filename);
		if (image_count == 1)  //读入第一张图片时获取图像宽高信息
		{
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
		}

		/* 提取角点 */
		if (0 == findChessboardCorners(imageInput, board_size, image_points_buf))
		{
			cout << "can not find chessboard corners!\n"; //找不到角点
			exit(1);
		}
		else
		{
			Mat view_gray;
			cvtColor(imageInput, view_gray, CV_RGB2GRAY);
			/* 亚像素精确化 */
			find4QuadCornerSubpix(view_gray, image_points_buf, Size(5, 5)); //对粗提取的角点进行精确化
			image_points_seq.push_back(image_points_buf);  //保存亚像素角点	
		}
	}
	int total = image_points_seq.size();
	cout << "total = " << total << endl;
	int CornerNum = board_size.width*board_size.height;  //每张图片上总的角点数
	cout << "image_size.width = " << image_size.width << endl;
	cout << "image_size.height = " << image_size.height << endl;
	cout << "Corner extraction completed" << endl;

	//以下是摄像机标定
	cout << "Start calibration………………" << endl;
	/*棋盘三维信息*/
	Size square_size = Size(10, 10);  /* 实际测量得到的标定板上每个棋盘格的大小 */
	vector<vector<Point3f> > object_points; /* 保存标定板上角点的三维坐标 */
										   /*内外参数*/
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 摄像机内参数矩阵 */
															/*
															事先标定好的右相机的内参矩阵
															fx 0 cx
															0 fy cy
															0 0  1
															*/
	vector<int> point_counts;  // 每幅图像中角点的数量
	Mat distCoeffs = Mat(5, 1, CV_32FC1, Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
	vector<Mat> tvecsMat;  /* 每幅图像的旋转向量 */
	vector<Mat> rvecsMat; /* 每幅图像的平移向量 */
						  /* 初始化标定板上角点的三维坐标 */
	int i, j, t;
	for (t = 0; t<image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i<board_size.height; i++)
		{
			for (j = 0; j<board_size.width; j++)
			{
				Point3f realPoint;
				/* 假设标定板放在世界坐标系中z=0的平面上 */
				realPoint.x = i * square_size.width;
				realPoint.y = j * square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	/* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
	for (i = 0; i<image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}
	/* 开始标定 */
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
	cout << "Calibration completed \n";
	//对标定结果进行评价
	cameraMatrixR = cameraMatrix;
	distCoeffR = distCoeffs;

	return 0;
}
int  outputCameraParam(void)
{
	/*保存数据*/
	/*输出数据*/
	FileStorage fs("intrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "cameraMatrixL" << cameraMatrixL << "cameraDistcoeffL" << distCoeffL << "cameraMatrixR" << cameraMatrixR << "cameraDistcoeffR" << distCoeffR;
		fs.release();
		cout << "cameraMatrixL=:" << cameraMatrixL << endl << "cameraDistcoeffL=:" << distCoeffL << endl << "cameraMatrixR=:" << cameraMatrixR << endl << "cameraDistcoeffR=:" << distCoeffR << endl;
	}
	else
	{
		cout << "Error: can not save the intrinsics!!!!!" << endl;
	}
	return 0;
}
int main()
{
	bdleft();
	bdright();
	outputCameraParam();
	waitKey(0);
	return 0;
}
