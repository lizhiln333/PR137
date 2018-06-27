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

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat R, T, E, F;
Mat Rl, Rr, Pl, Pr, Q;                                  //У����ת����R��ͶӰ����P ��ͶӰ����Q (�����о���ĺ�����ͣ�   
Mat mapLx, mapLy, mapRx, mapRy;                         //ӳ���  
Rect validROIL, validROIR;                              //ͼ��У��֮�󣬻��ͼ����вü��������validROI����ָ�ü�֮�������  

Mat cameraMatrixL = Mat_<double>(3, 3);
Mat distCoeffL = Mat_<double>(5, 1);
Mat cameraMatrixR = Mat_<double>(3, 3);
Mat distCoeffR = Mat_<double>(5, 1);

int imageWidth = 1280;                             //����ͷ�ķֱ���  
int imageHeight = 960;

Size imageSize = Size(imageWidth, imageHeight);



int inputCameraParam(void)
{
	/*��ȡ����*/

	FileStorage fs("intrinsics2.yml", FileStorage::READ);
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
	fs.open("extrinsics.yml", FileStorage::READ);
	if (fs.isOpened())
	{
		fs["R"] >> R;
		fs["T"] >> T;
		fs["Rl"] >> Rl;
		fs["Rr"] >> Rr;
		fs["Pl"] >> Pl;
		fs["Pr"] >> Pr;
		fs["Q"] >> Q;
		fs["E"] >> E;
		fs["F"] >> F;
		//cout << "R=" << R << endl << "T=" << T << endl << "Rl=" << Rl << endl << "Rr=" << Rr << endl << "Pl=" << Pl << endl << "Pr=" << Pr << endl << "Q=" << Q << endl << "E=" << E << endl << "F=" << F << endl;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";
	return 0;
}
int remap()
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
	mapx,mapy������ӳ�����������Ը�remap()�������ã���У��ͼ��ʹ������ͼ���沢���ж�׼
	ininUndistortRectifyMap()�Ĳ���newCameraMatrix����У����������������openCV���棬У����ļ��������Mrect�Ǹ�ͶӰ����Pһ�𷵻صġ�
	�������������ﴫ��ͶӰ����P���˺������Դ�ͶӰ����P�ж���У��������������
	*/
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);


	Mat rectifyImageL, rectifyImageR;
	Mat leftImage, rightImage;
	/*******��ȡ���������ͼƬ*******/
	leftImage = imread("left06.jpg", CV_LOAD_IMAGE_COLOR);
	rightImage = imread("right06.jpg", CV_LOAD_IMAGE_COLOR);
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
	imwrite("rectifyImageL.png", rectifyImageL);
	imwrite("rectifyImageR.png", rectifyImageR);
	/*���沢�������*/


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
	//waitKey(0);
	//system("pause");
	return 0;
}
int shicha(void)
{

	cout << "shicha" << endl;
	IplImage* pImgleft = cvLoadImage("rectifyImageL.png", 0);      //��ȡ��ͼ�� 
	cv::Mat left = cv::cvarrToMat(pImgleft);
	CvMat* LEFT = cvCreateMat(pImgleft->height, pImgleft->width, CV_8UC1);
	cvConvert(pImgleft, LEFT);
	IplImage* pImgright = cvLoadImage("rectifyImageR.png", 0);      //��ȡ��ͼ��
	cv::Mat right = cv::cvarrToMat(pImgright);
	CvMat* RIGHT = cvCreateMat(pImgright->height, pImgright->width, CV_8UC1);
	cvConvert(pImgright, RIGHT);
	/***ͼ���ȡ***/

	int ND = (((pImgleft->width) / 8) + 15) & -16;
	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);
	sgbm->setPreFilterCap(63);
	//int SADWindowSize = 3;              
	int SADWindowSize = 5;        //SAD���ڴ�С3 , 5 , 7 , 9 , 11
								  //int cn = 1;
	int cn = pImgleft->nChannels;    //��ȡͼ��ͨ����
	sgbm->setMode(3);              //ģʽѡ��
	sgbm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 3);
	//int cm = sgbm->getBlockSize();
	sgbm->setP1(8 * cn * SADWindowSize * SADWindowSize);//8
	sgbm->setP2(32 * cn * SADWindowSize * SADWindowSize);
	sgbm->setMinDisparity(0);             //��С�Ӳ�     
	sgbm->setNumDisparities(ND);            //�Ӳ�ڣ�������Ӳ�ֵ����С�Ӳ�ֵ֮��,

	sgbm->setUniquenessRatio(10);		//�Ӳ�Ψһ�԰ٷֱ�

	sgbm->setSpeckleWindowSize(32);        //����Ӳ���ͨ����仯�ȵĴ��ڴ�С
										   //sgbm->setSpeckleRange(32);             //�Ӳ�仯��ֵ
	sgbm->setSpeckleRange(32);
	sgbm->setDisp12MaxDiff(1);


	Mat disp, disp8;

	int64 t = getTickCount();
	sgbm->compute(left, right, disp);
	t = getTickCount() - t;
	cout << "Time elapsed:" << t * 1000 / getTickFrequency() << endl;
	disp.convertTo(disp8, CV_8U, 255 / (ND*16.));


	namedWindow("left", 1);
	cvShowImage("left", LEFT);
	namedWindow("right", 1);
	cvShowImage("right", RIGHT);
	namedWindow("disparity", 1);
	imshow("disparity", disp);
	/********�Ӳ�ͼ��ʾ*******/
	imwrite("sgbm_disparity.png", disp8);
	imwrite("sgbm_disparity1.png", disp);
	//imwrite(outname, disp8);
	waitKey(0);
	cvDestroyAllWindows();
	cout << "shicha successful" << endl;
	return 0;
}
int main()
{
	inputCameraParam();
	remap();
	shicha();
	waitKey(0);
	return 0;
}