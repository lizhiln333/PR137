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
���ȱ궨�õ���������ڲξ���
fx 0 cx
0 fy cy
0 0  1
*/
Mat cameraMatrixR = Mat_<double>(3, 3);
Mat distCoeffR = Mat_<double>(5, 1);
int bdleft()
{
	ifstream fin("calibdata_left.txt"); /* �궨����ͼ���ļ���·�� */
										//��ȡÿһ��ͼ�񣬴�����ȡ���ǵ㣬Ȼ��Խǵ���������ؾ�ȷ��	
	cout << "Start to extract corner points������������" << endl;
	int image_count = 0;  /* ͼ������ */
	Size image_size;  /* ͼ��ĳߴ� */
	Size board_size = Size(11, 8);    /* �궨����ÿ�С��еĽǵ��� */
	vector<Point2f> image_points_buf;  /* ����ÿ��ͼ���ϼ�⵽�Ľǵ� */
	vector<vector<Point2f> > image_points_seq; /* �����⵽�����нǵ� */
	string filename;
	int count = -1;//���ڴ洢�ǵ������
	while (getline(fin, filename))
	{
		image_count++;
		// ���ڹ۲�������
		cout << "image_count = " << image_count << endl;
		/* �������*/
		//cout << "-->count = " << count;
		Mat imageInput = imread(filename);
		if (image_count == 1)  //�����һ��ͼƬʱ��ȡͼ������Ϣ
		{
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;

		}

		/* ��ȡ�ǵ� */
		if (0 == findChessboardCorners(imageInput, board_size, image_points_buf))
		{
			cout << "can not find chessboard corners!\n"; //�Ҳ����ǵ�
			exit(1);
		}
		else
		{
			Mat view_gray;
			cvtColor(imageInput, view_gray, CV_RGB2GRAY);
			/* �����ؾ�ȷ�� */
			find4QuadCornerSubpix(view_gray, image_points_buf, Size(5, 5)); //�Դ���ȡ�Ľǵ���о�ȷ��
																			//cout << image_points_buf << endl;
			image_points_seq.push_back(image_points_buf);  //���������ؽǵ�	
		}
	}
	int total = image_points_seq.size();
	cout << "total = " << total << endl;
	int CornerNum = board_size.width*board_size.height;  //ÿ��ͼƬ���ܵĽǵ�
	cout << CornerNum << endl;
	cout << "image_size.width = " << image_size.width << endl;
	cout << "image_size.height = " << image_size.height << endl;
	cout << "Corner extraction completed" << endl;

	//������������궨
	cout << "Start calibration������������" << endl;
	/*������ά��Ϣ*/
	Size square_size = Size(20, 20);  /* ʵ�ʲ����õ��ı궨����ÿ�����̸�Ĵ�С */
	vector<vector<Point3f> > object_points; /* ����궨���Ͻǵ����ά���� */
										   /*�������*/
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* ������ڲ������� */
															/*
															���ȱ궨�õ���������ڲξ���
															fx 0 cx
															0 fy cy
															0 0  1
															*/
	vector<int> point_counts;  // ÿ��ͼ���нǵ������
	Mat distCoeffs = Mat(5, 1, CV_32FC1, Scalar::all(0)); /* �������5������ϵ����k1,k2,p1,p2,k3 */
	vector<Mat> tvecsMat;  /* ÿ��ͼ�����ת���� */
	vector<Mat> rvecsMat; /* ÿ��ͼ���ƽ������ */
						  /* ��ʼ���궨���Ͻǵ����ά���� */
	int i, j, t;
	for (t = 0; t<image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i<board_size.height; i++)
		{
			for (j = 0; j<board_size.width; j++)
			{
				Point3f realPoint;
				/* ����궨�������������ϵ��z=0��ƽ���� */
				realPoint.x = i * square_size.width;
				realPoint.y = j * square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	/* ��ʼ��ÿ��ͼ���еĽǵ��������ٶ�ÿ��ͼ���ж����Կ��������ı궨�� */
	for (i = 0; i<image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}
	/* ��ʼ�궨 */
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
	cout << "Calibration completed\n";
	//�Ա궨�����������
	cameraMatrixL = cameraMatrix;
	distCoeffL = distCoeffs;

	return 0;
}
int bdright()
{
	ifstream fin("calibdata_right.txt"); /* �궨����ͼ���ļ���·�� */
	cout << "Start to extract corner points������������" << endl;
	int image_count = 0;  /* ͼ������ */
	Size image_size;  /* ͼ��ĳߴ� */
	Size board_size = Size(11, 8);    /* �궨����ÿ�С��еĽǵ��� */
	vector<Point2f> image_points_buf;  /* ����ÿ��ͼ���ϼ�⵽�Ľǵ� */
	vector<vector<Point2f> > image_points_seq; /* �����⵽�����нǵ� */
	string filename;
	int count = -1;//���ڴ洢�ǵ������
	while (getline(fin, filename))
	{
		image_count++;
		// ���ڹ۲�������
		cout << "image_count = " << image_count << endl;
		/* �������*/
		//cout << "-->count = " << count;
		Mat imageInput = imread(filename);
		if (image_count == 1)  //�����һ��ͼƬʱ��ȡͼ������Ϣ
		{
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
		}

		/* ��ȡ�ǵ� */
		if (0 == findChessboardCorners(imageInput, board_size, image_points_buf))
		{
			cout << "can not find chessboard corners!\n"; //�Ҳ����ǵ�
			exit(1);
		}
		else
		{
			Mat view_gray;
			cvtColor(imageInput, view_gray, CV_RGB2GRAY);
			/* �����ؾ�ȷ�� */
			find4QuadCornerSubpix(view_gray, image_points_buf, Size(5, 5)); //�Դ���ȡ�Ľǵ���о�ȷ��
			image_points_seq.push_back(image_points_buf);  //���������ؽǵ�	
		}
	}
	int total = image_points_seq.size();
	cout << "total = " << total << endl;
	int CornerNum = board_size.width*board_size.height;  //ÿ��ͼƬ���ܵĽǵ���
	cout << "image_size.width = " << image_size.width << endl;
	cout << "image_size.height = " << image_size.height << endl;
	cout << "Corner extraction completed" << endl;

	//������������궨
	cout << "Start calibration������������" << endl;
	/*������ά��Ϣ*/
	Size square_size = Size(10, 10);  /* ʵ�ʲ����õ��ı궨����ÿ�����̸�Ĵ�С */
	vector<vector<Point3f> > object_points; /* ����궨���Ͻǵ����ά���� */
										   /*�������*/
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* ������ڲ������� */
															/*
															���ȱ궨�õ���������ڲξ���
															fx 0 cx
															0 fy cy
															0 0  1
															*/
	vector<int> point_counts;  // ÿ��ͼ���нǵ������
	Mat distCoeffs = Mat(5, 1, CV_32FC1, Scalar::all(0)); /* �������5������ϵ����k1,k2,p1,p2,k3 */
	vector<Mat> tvecsMat;  /* ÿ��ͼ�����ת���� */
	vector<Mat> rvecsMat; /* ÿ��ͼ���ƽ������ */
						  /* ��ʼ���궨���Ͻǵ����ά���� */
	int i, j, t;
	for (t = 0; t<image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i<board_size.height; i++)
		{
			for (j = 0; j<board_size.width; j++)
			{
				Point3f realPoint;
				/* ����궨�������������ϵ��z=0��ƽ���� */
				realPoint.x = i * square_size.width;
				realPoint.y = j * square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	/* ��ʼ��ÿ��ͼ���еĽǵ��������ٶ�ÿ��ͼ���ж����Կ��������ı궨�� */
	for (i = 0; i<image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}
	/* ��ʼ�궨 */
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
	cout << "Calibration completed \n";
	//�Ա궨�����������
	cameraMatrixR = cameraMatrix;
	distCoeffR = distCoeffs;

	return 0;
}
int  outputCameraParam(void)
{
	/*��������*/
	/*�������*/
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
