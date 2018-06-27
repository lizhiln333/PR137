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
Mat Rl, Rr, Pl, Pr, Q;                                  //校正旋转矩阵R，投影矩阵P 重投影矩阵Q (下面有具体的含义解释）   
Mat mapLx, mapLy, mapRx, mapRy;                         //映射表  
Rect validROIL, validROIR;                              //图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  

Mat cameraMatrixL = Mat_<double>(3, 3);
Mat distCoeffL = Mat_<double>(5, 1);
Mat cameraMatrixR = Mat_<double>(3, 3);
Mat distCoeffR = Mat_<double>(5, 1);

int imageWidth = 1280;                             //摄像头的分辨率  
int imageHeight = 960;

Size imageSize = Size(imageWidth, imageHeight);



int inputCameraParam(void)
{
	/*读取数据*/

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
	立体校正的时候需要两幅图像共面并且行对准 以使得立体匹配更加的可靠
	使得两幅图像共面的方法就是把两个摄像头的图像投影到一个公共成像面上，这样每幅图像从本图像平面投影到公共图像平面都需要一个旋转矩阵R
	stereoRectify 这个函数计算的就是从图像平面投影都公共成像平面的旋转矩阵Rl,Rr。 Rl,Rr即为左右相机平面行对准的校正旋转矩阵。
	左相机经过Rl旋转，右相机经过Rr旋转之后，两幅图像就已经共面并且行对准了。
	其中Pl,Pr为两个相机的投影矩阵，其作用是将3D点的坐标转换到图像的2D点的坐标:P*[X Y Z 1]' =[x y w]
	Q矩阵为重投影矩阵，即矩阵Q可以把2维平面(图像平面)上的点投影到3维空间的点:Q*[x y d 1] = [X Y Z W]。其中d为左右两幅图像的时差
	*/
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q,
		CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL, &validROIR);
	/*
	根据stereoRectify 计算出来的R 和 P 来计算图像的映射表 mapx,mapy
	mapx,mapy这两个映射表接下来可以给remap()函数调用，来校正图像，使得两幅图像共面并且行对准
	ininUndistortRectifyMap()的参数newCameraMatrix就是校正后的摄像机矩阵。在openCV里面，校正后的计算机矩阵Mrect是跟投影矩阵P一起返回的。
	所以我们在这里传入投影矩阵P，此函数可以从投影矩阵P中读出校正后的摄像机矩阵
	*/
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);


	Mat rectifyImageL, rectifyImageR;
	Mat leftImage, rightImage;
	/*******读取常规带矫正图片*******/
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
	经过remap之后，左右相机的图像已经共面并且行对准了
	*/
	remap(rectifyImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
	remap(rectifyImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

	imshow("ImageL", rectifyImageL);
	imshow("ImageR", rectifyImageR);
	imwrite("rectifyImageL.png", rectifyImageL);
	imwrite("rectifyImageR.png", rectifyImageR);
	/*保存并输出数据*/


	/*
	把校正结果显示出来
	把左右两幅图像显示到同一个画面上
	这里只显示了最后一副图像的校正结果。并没有把所有的图像都显示出来
	*/
	Mat canvas;
	double sf;
	int w, h;
	sf = 600. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width * sf);
	h = cvRound(imageSize.height * sf);
	canvas.create(h, w * 2, CV_8UC3);

	/*左图像画到画布上*/
	Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
	resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
	Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域    
		cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
	rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  

	cout << "Painted ImageL" << endl;

	/*右图像画到画布上*/
	canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
	resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
		cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
	rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);

	cout << "Painted ImageR" << endl;

	/*画上对应的线条*/
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
	IplImage* pImgleft = cvLoadImage("rectifyImageL.png", 0);      //读取主图像 
	cv::Mat left = cv::cvarrToMat(pImgleft);
	CvMat* LEFT = cvCreateMat(pImgleft->height, pImgleft->width, CV_8UC1);
	cvConvert(pImgleft, LEFT);
	IplImage* pImgright = cvLoadImage("rectifyImageR.png", 0);      //读取副图像
	cv::Mat right = cv::cvarrToMat(pImgright);
	CvMat* RIGHT = cvCreateMat(pImgright->height, pImgright->width, CV_8UC1);
	cvConvert(pImgright, RIGHT);
	/***图像读取***/

	int ND = (((pImgleft->width) / 8) + 15) & -16;
	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);
	sgbm->setPreFilterCap(63);
	//int SADWindowSize = 3;              
	int SADWindowSize = 5;        //SAD窗口大小3 , 5 , 7 , 9 , 11
								  //int cn = 1;
	int cn = pImgleft->nChannels;    //获取图像通道数
	sgbm->setMode(3);              //模式选择
	sgbm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 3);
	//int cm = sgbm->getBlockSize();
	sgbm->setP1(8 * cn * SADWindowSize * SADWindowSize);//8
	sgbm->setP2(32 * cn * SADWindowSize * SADWindowSize);
	sgbm->setMinDisparity(0);             //最小视差     
	sgbm->setNumDisparities(ND);            //视差窗口，即最大视差值与最小视差值之差,

	sgbm->setUniquenessRatio(10);		//视差唯一性百分比

	sgbm->setSpeckleWindowSize(32);        //检查视差连通区域变化度的窗口大小
										   //sgbm->setSpeckleRange(32);             //视差变化阈值
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
	/********视差图显示*******/
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