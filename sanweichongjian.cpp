//by shuishui shiwenjun 20160926
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>  
#include <pcl/io/io.h>  
#include <pcl/io/pcd_io.h>  
#include <opencv2/opencv.hpp>  

using namespace cv;
using namespace std;
using namespace pcl;

int user_data;
//相机内参，根据输入改动
const double u0 = 653.66771864959412;//由于后面resize成原图的1/4所以有些参数要缩小相同倍数
const double v0 = 489.55058641617688;
const double fx = 1421.08765951443586;
const double fy = 1421.7724298090397;
const double Tx = 71.449468540383137;
const double doffs = 350;

void viewerOneOff(visualization::PCLVisualizer& viewer)
{
	viewer.setBackgroundColor(0.0, 0.0, 0.0);
}

int main()
{
	PointCloud<PointXYZRGB> cloud_a;
	PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);



	Mat color1 = imread("right31.jpg");
	Mat depth = imread("31 (1).png");
	////Resize
	//color1.resize();
	Mat color;
	resize(color1, color, Size(color1.cols , color1.rows ), 0, 0, CV_INTER_LINEAR);
	//imshow("h",color);
	//waitKey(0);

	int rowNumber = color.rows;
	int colNumber = color.cols;

	cloud_a.height = rowNumber;
	cloud_a.width = colNumber;
	cloud_a.points.resize(cloud_a.width * cloud_a.height);

	for (unsigned int u = 0; u < rowNumber; ++u)
	{
		for (unsigned int v = 0; v < colNumber; ++v)
		{
			/*unsigned int num = rowNumber*colNumber-(u*colNumber + v)-1;*/
			unsigned int num = u*colNumber + v;
			double Xw = 0, Yw = 0, Zw = 0;


			Zw = fx*Tx / (((double)depth.at<Vec3b>(u, v)[0]) + doffs);
			Xw = (v + 1 - u0) * Zw / fx;
			Yw = (u + 1 - v0) * Zw / fy;

			cloud_a.points[num].b = color.at<Vec3b>(u, v)[0];
			cloud_a.points[num].g = color.at<Vec3b>(u, v)[1];
			cloud_a.points[num].r = color.at<Vec3b>(u, v)[2];

			cloud_a.points[num].x = Xw;
			cloud_a.points[num].y = Yw;
			cloud_a.points[num].z = Zw;
		}
	}

	*cloud = cloud_a;

	visualization::CloudViewer viewer("Cloud Viewer");

	viewer.showCloud(cloud);

	viewer.runOnVisualizationThreadOnce(viewerOneOff);

	while (!viewer.wasStopped())
	{
		user_data = 9;
	}

	return 0;
}