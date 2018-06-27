//#include "stdafx.h"  
//#include <opencv2/opencv.hpp>  
//#include "opencv2/highgui.hpp"
//#include "cv.h"  
//#include <cv.hpp>  
//#include <iostream>  

#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>


using namespace std;
using namespace cv;

int imageWidth = 1280;                             //����ͷ�ķֱ���  
int imageHeight = 960;
const int boardWidth = 11;                               //����Ľǵ���Ŀ  
const int boardHeight = 8;                              //����Ľǵ�����  
const int boardCorner = boardWidth * boardHeight;       //�ܵĽǵ�����  
const int frameNumber = 25;                             //����궨ʱ��Ҫ���õ�ͼ��֡��  
const int squareSize = 20;                              //�궨��ڰ׸��ӵĴ�С ��λmm  
const Size boardSize = Size(boardWidth, boardHeight);   //  


Mat R, T, E, F;                                         //R ��תʸ�� Tƽ��ʸ�� E�������� F��������  
vector<Mat> rvecs;                                        //��ת����  
vector<Mat> tvecs;                                        //ƽ������  
vector<vector<Point2f> > imagePointL;                    //��������������Ƭ�ǵ�����꼯��  
vector<vector<Point2f> > imagePointR;                    //�ұ������������Ƭ�ǵ�����꼯��  
vector<vector<Point3f> > objRealPoint;                   //����ͼ��Ľǵ��ʵ���������꼯��  


vector<Point2f> cornerL;                              //��������ĳһ��Ƭ�ǵ����꼯��  
vector<Point2f> cornerR;                              //�ұ������ĳһ��Ƭ�ǵ����꼯��  

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;

Mat Rl, Rr, Pl, Pr, Q;                                  //У����ת����R��ͶӰ����P ��ͶӰ����Q (�����о���ĺ�����ͣ�   
Mat mapLx, mapLy, mapRx, mapRy;                         //ӳ���  
Rect validROIL, validROIR;                              //ͼ��У��֮�󣬻��ͼ����вü��������validROI����ָ�ü�֮�������  


														//���ȱ궨�õ���������ڲξ���
														//fx 0 cx
														//0 fy cy
														//0 0  1
														//*/

														/******���뵥Ŀ�궨���*******/
Mat cameraMatrixL = Mat_<double>(3, 3);
Mat distCoeffL = Mat_<double>(5, 1);
Mat cameraMatrixR = Mat_<double>(3, 3);
Mat distCoeffR = Mat_<double>(5, 1);

/*����궨����ģ���ʵ����������*/
int calRealPoint(vector<vector<Point3f> >& obj, int boardwidth, int boardheight, int imgNumber, int squaresize)
{
	//  Mat imgpoint(boardheight, boardwidth, CV_32FC3,Scalar(0,0,0));  
	vector<Point3f> imgpoint;
	for (int rowIndex = 0; rowIndex < boardheight; rowIndex++)
	{
		for (int colIndex = 0; colIndex < boardwidth; colIndex++)
		{
			//  imgpoint.at<Vec3f>(rowIndex, colIndex) = Vec3f(rowIndex * squaresize, colIndex*squaresize, 0);  
			imgpoint.push_back(Point3f(rowIndex * squaresize, colIndex * squaresize, 0));
		}
	}
	for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
	{
		obj.push_back(imgpoint);
	}
	return 0;
}

int outputCameraParam(void)
{
	/*��������*/
	/*�������*/
	FileStorage fs("intrinsics2.yml", FileStorage::WRITE);
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

	fs.open("extrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "Rl" << Rl << "Rr" << Rr << "Pl" << Pl << "Pr" << Pr << "Q" << Q << "E" << E << "F" << F;
		cout << "R=" << R << endl << "T=" << T << endl << "Rl=" << Rl << endl << "Rr=" << Rr << endl << "Pl=" << Pl << endl << "Pr=" << Pr << endl << "Q=" << Q << endl << "E=" << E << endl << "F=" << F << endl;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";
	return 0;
}
int inputCameraParam(void)
{
	/*��������*/
	/*�������*/
	FileStorage fs("intrinsics.yml", FileStorage::READ);
	if (fs.isOpened())
	{
		fs["cameraMatrixL"] >> cameraMatrixL;
		fs["cameraDistcoeffL"] >> distCoeffL;
		fs["cameraMatrixR"] >> cameraMatrixR;
		fs["cameraDistcoeffR"] >> distCoeffR;
		fs.release();
		cout << "cameraMatrixL=:" << cameraMatrixL << endl << "cameraDistcoeffL=:" << distCoeffL << endl << "cameraMatrixR=:" << cameraMatrixR << endl << "cameraDistcoeffR=:" << distCoeffR << endl;
	}
	else
	{
		cout << "Error: can not save the intrinsics!!!!!" << endl;
	}
	return 0;
}
int inputImge(void)
{
	Size image_size;
	Mat imageInput = imread("left01.jpg");
	imageWidth = imageInput.cols;
	imageHeight = imageInput.rows;
	return 0;
}
int remap(Size imageSize)
{
	/*
	����У����ʱ����Ҫ����ͼ���沢���ж�׼ ��ʹ������ƥ����ӵĿɿ�
	ʹ������ͼ����ķ������ǰ���������ͷ��ͼ��ͶӰ��һ�������������ϣ�����ÿ��ͼ��ӱ�ͼ��ƽ��ͶӰ������ͼ��ƽ�涼��Ҫһ����ת����R
	stereoRectify �����������ľ��Ǵ�ͼ��ƽ��ͶӰ����������ƽ�����ת����Rl,Rr�� Rl,Rr��Ϊ�������ƽ���ж�׼��У����ת����
	���������Rl��ת�����������Rr��ת֮������ͼ����Ѿ����沢���ж�׼�ˡ�
	����Pl,PrΪ���������ͶӰ�����������ǽ�3D�������ת����ͼ���2D�������:P*[X Y Z 1]' =[x y w]
	Q����Ϊ��ͶӰ���󣬼�����Q���԰�2άƽ��(ͼ��ƽ��)�ϵĵ�ͶӰ��3ά�ռ�ĵ�:Q*[x y d 1] = [X Y Z W]������dΪ��������ͼ���ʱ��
	*/
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q,
		CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL, &validROIR);
	/*
	����stereoRectify ���������R �� P ������ͼ���ӳ��� mapx,mapy
	mapx,mapy������ӳ������������Ը�remap()�������ã���У��ͼ��ʹ������ͼ���沢���ж�׼
	ininUndistortRectifyMap()�Ĳ���newCameraMatrix����У����������������openCV���棬У����ļ��������Mrect�Ǹ�ͶӰ����Pһ�𷵻صġ�
	�������������ﴫ��ͶӰ����P���˺������Դ�ͶӰ����P�ж���У��������������
	*/
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);


	Mat rectifyImageL, rectifyImageR;
	Mat leftImage, rightImage;
	/*******��ȡ���������ͼƬ*******/
	leftImage = imread("left26.jpg", CV_LOAD_IMAGE_COLOR);
	rightImage = imread("right26.jpg", CV_LOAD_IMAGE_COLOR);
	cvtColor(leftImage, grayImageL, CV_BGR2GRAY);
	cvtColor(rightImage, grayImageR, CV_BGR2GRAY);
	cvtColor(grayImageL, rectifyImageL, CV_GRAY2BGR);
	cvtColor(grayImageR, rectifyImageR, CV_GRAY2BGR);

	imshow("Rectify Before", rectifyImageL);
	//Mat rectifyImageL, rectifyImageR;
	//cvtColor(grayImageL, rectifyImageL, CV_GRAY2BGR);
	//cvtColor(grayImageR, rectifyImageR, CV_GRAY2BGR);
	/*
	����remap֮�����������ͼ���Ѿ����沢���ж�׼��
	*/
	remap(rectifyImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
	remap(rectifyImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

	imshow("ImageL", rectifyImageL);
	imshow("ImageR", rectifyImageR);




	/*
	��У�������ʾ����
	����������ͼ����ʾ��ͬһ��������
	����ֻ��ʾ�����һ��ͼ���У���������û�а����е�ͼ����ʾ����
	*/
	Mat canvas;
	double sf;
	int w, h;
	sf = 600. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width * sf);
	h = cvRound(imageSize.height * sf);
	canvas.create(h, w * 2, CV_8UC3);

	/*��ͼ�񻭵�������*/
	Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //�õ�������һ����  
	resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //��ͼ�����ŵ���canvasPartһ����С  
	Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //��ñ���ȡ������    
		cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
	rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //����һ������  

	cout << "Painted ImageL" << endl;

	/*��ͼ�񻭵�������*/
	canvasPart = canvas(Rect(w, 0, w, h));                                      //��û�������һ����  
	resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
		cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
	rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);

	cout << "Painted ImageR" << endl;

	/*���϶�Ӧ������*/
	for (int i = 0; i < canvas.rows; i += 16)
		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);

	imshow("rectified", canvas);

	cout << "wait key" << endl;
	waitKey(0);
	system("pause");
	return 0;
}
int main()
{
	Mat img;
	int goodFrameCount = 0;
	namedWindow("ImageL");
	namedWindow("ImageR");
	cout << "Press Q to exit ..." << endl;
	inputCameraParam();
	inputImge();
	cout << "Single target information has been read" << endl;
	Size imageSize = Size(imageWidth, imageHeight);
	while (goodFrameCount < frameNumber)
	{
		char filename[100];
		/*��ȡ��ߵ�ͼ��*/
		sprintf(filename, "left%02d.jpg", goodFrameCount + 1);
		rgbImageL = imread(filename, CV_LOAD_IMAGE_COLOR);
		cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);

		/*��ȡ�ұߵ�ͼ��*/
		sprintf(filename, "right%02d.jpg", goodFrameCount + 1);
		rgbImageR = imread(filename, CV_LOAD_IMAGE_COLOR);
		cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);

		bool isFindL, isFindR;

		isFindL = findChessboardCorners(rgbImageL, boardSize, cornerL);
		isFindR = findChessboardCorners(rgbImageR, boardSize, cornerR);
		if (isFindL == true && isFindR == true)  //�������ͼ���ҵ������еĽǵ� ��˵��������ͼ���ǿ��е�  
		{
			/*
			Size(5,5) �������ڵ�һ���С
			Size(-1,-1) ������һ��ߴ�
			TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1)������ֹ����
			*/
			cornerSubPix(grayImageL, cornerL, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
			drawChessboardCorners(rgbImageL, boardSize, cornerL, isFindL);
			//	imshow("chessboardL", rgbImageL);
			imagePointL.push_back(cornerL);


			cornerSubPix(grayImageR, cornerR, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
			drawChessboardCorners(rgbImageR, boardSize, cornerR, isFindR);
			//	imshow("chessboardR", rgbImageR);
			imagePointR.push_back(cornerR);

			/*
			����Ӧ���ж�������ͼ���ǲ��Ǻõģ��������ƥ��Ļ��ſ��������궨
			������������̵��У��õ�ͼ����ϵͳ�Դ���ͼ�񣬶��ǿ���ƥ��ɹ��ġ�
			���������û���ж�
			*/
			//string filename = "res\\image\\calibration";  
			//filename += goodFrameCount + ".jpg";  
			//cvSaveImage(filename.c_str(), &IplImage(rgbImage));       //�Ѻϸ��ͼƬ��������  
			goodFrameCount++;
			cout << "The image is good" << endl;
		}
		else
		{
			cout << "The image is bad please try again" << endl;
		}

		if (waitKey(10) == 'q')
		{
			break;
		}
	}

	/*
	����ʵ�ʵ�У�������ά����
	����ʵ�ʱ궨���ӵĴ�С������
	*/
	calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
	cout << "cal real successful" << endl;

	/*
	�궨����ͷ
	��������������ֱ𶼾����˵�Ŀ�궨
	�����ڴ˴�ѡ��flag = CALIB_USE_INTRINSIC_GUESS
	*/
	double rms = stereoCalibrate(objRealPoint, imagePointL, imagePointR,
		cameraMatrixL, distCoeffL,
		cameraMatrixR, distCoeffR,
		Size(imageWidth, imageHeight), R, T, E, F,
		CALIB_USE_INTRINSIC_GUESS,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
	outputCameraParam();
	remap(imageSize);

	return 0;
}