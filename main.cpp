#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <iostream>

using namespace std;
using namespace cv;

Mat localRadon(Mat edgeImg, int theta, int winsz) {
	int nrows = edgeImg.rows;
	int ncols = edgeImg.cols;
	int nrho = ceil(sqrt(nrows*nrows + ncols * ncols));
	int nt = nrho;
	int len = countNonZero(edgeImg);
	int *x = new int[len];
	int *y = new int[len];
	//[y, x] = find(bimg > 0)
	int q = 0;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			if (edgeImg.at<uchar>(i, j) == 255) {
				y[q] = i;//行
				x[q] = j;//列
				q++;
			}
		}
	}
	Mat cube(nrho, nt, CV_8UC1, Scalar::all(0));
	for (int n = 0; n < q; n++) {
		double t = (x[n] - ncols / 2.0)*sin(theta) - (y[n] - nrows / 2.0)*cos(theta) + nt / 2;
		double rho = (x[n] - ncols / 2.0)*cos(theta) - (y[n] - nrows / 2.0)*sin(theta) + nrho / 2;
		cube.at<uchar>(round(rho), round(t)) = 1;
	}
	cout << (int)cube.at<uchar>(31, 482) << endl;
	for (int j = 0; j < nt; j++) {
		for (int k = 1; k < nrho; k++)
			cube.at<uchar>(k, j) += cube.at<uchar>(k - 1, j);
	}
	cout << (int)cube.at<uchar>(200, 200) << endl;
	for (int j = 0; j < nt; j++) {
		for (int k = 0; k < nrho; k++) {
			int rho1 = (k + winsz) < (nrho - 1) ? (k + winsz) : (nrho - 1);
			int rho2 = (k - winsz) > 0 ? (k - winsz) : 0;
			int sum = cube.at<uchar>(rho1, j) - cube.at<uchar>(rho2, j);
			cube.at<uchar>(k, j) = sum * sum*sum;
		}
	}
	cout << (int)cube.at<uchar>(200, 200) << endl;
	for (int j = 0; j < nrho; j++) {
		for (int k = 1; k < nt; k++)
			cube.at<uchar>(j, k) += cube.at<uchar>(j, k - 1);
	}
	cout << (int)cube.at<uchar>(200, 200) << endl;
	for (int j = 0; j < nrho; j++) {
		for (int k = 0; k < nt; k++) {
			int t1 = (k + round(winsz / 2.0)) < (nt - 1) ? (k + round(winsz / 2.0)) : (nt - 1);
			int t2 = (k - round(winsz / 2.0)) > 0 ? (k - round(winsz / 2.0)) : 0;
			cube.at<uchar>(j, k) = cube.at<uchar>(j, t1) - cube.at<uchar>(j, t2);
		}
	}
	cout << (int)cube.at<uchar>(200, 200) << endl;
	Mat pcube(nrows, ncols, CV_8UC1, Scalar::all(0));
	for (int i = 1; i < nrows - 1; i++) {
		for (int j = 1; j < ncols - 1; j++) {
			double t = (j - ncols / 2.0)*sin(theta) - (i - nrows / 2.0)*cos(theta) + nt / 2;
			double rho = (j - ncols / 2.0)*cos(theta) + (i - nrows / 2.0)*sin(theta) + nrho / 2;
			int divide = pow(winsz, 4)*nrows;
			pcube.at<uchar>(i, j) = saturate_cast<uchar>(cube.at<uchar>(rho, t) / divide);
		}
	}
	cout << (int)cube.at<uchar>(200, 200) << endl;
	delete[]x;
	delete[]y;
	return pcube;
}

int main() {
	Mat src = imread("D:/vs work/1.jpg");
	cout << src.channels() << endl;
	cout << (int)src.at<Vec3b>(200, 300)[0] << endl;
	cout << src.type() << endl;
	Mat edge;
	edge.create(src.size(), src.type());
	Canny(src, edge, 200, 100, 3);
	cout << (int)edge.at<uchar>(31, 482) << endl;
	cout << edge.channels() << endl;
	cout << edge.type() << endl;

	double theta = 60 / 180.0*3.1415926;
	Mat dst = localRadon(edge, theta, 20);
	imshow("radon.png", dst);
	/// 等待用户按键
	waitKey();
	return 0;
}



/*#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <iostream>

using namespace std;
using namespace cv;

Mat localRadon(Mat edgeImg, int theta, int winsz) {
	int nrows = edgeImg.rows;
	int ncols = edgeImg.cols;
	int nrho = ceil(sqrt(nrows*nrows + ncols * ncols));
	int nt = nrho;
	int len = countNonZero(edgeImg);
	int *x = new int[len];
	int *y = new int[len];
	//[y, x] = find(bimg > 0)
	int q = 0;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			if (edgeImg.at<uchar>(i, j) == 255) {
				y[q] = i;//行
				x[q] = j;//列
				q++;
			}
		}
	}
	Mat cube(nrho, nt, CV_8UC1, Scalar::all(0));
	/*for (int n = 0; n < q; n++) {
		double t = (x[n] - ncols / 2.0)*sin(theta) - (y[n] - nrows / 2.0)*cos(theta) + nt / 2;
		double rho = (x[n] - ncols / 2.0)*cos(theta) - (y[n] - nrows / 2.0)*sin(theta) + nrho / 2;
		int i = x[n], j = y[n];
		cube.at<uchar>(round(rho), round(t)) = edgeImg.at<uchar>(i,j);
	}*/
	/*for (int y = 0; y <nrows; y++) {
		for (int x = 0; x < ncols; x++) {
			double t = (x - ncols / 2.0) * sin(theta) - (y - nrows / 2.0)*cos(theta) + nt / 2;
			double rho = (x - ncols / 2.0) * cos(theta) + (y - nrows / 2.0) * sin(theta) + nrho / 2;
			cube.at<uchar>(round(rho), round(t)) = 1;
			//cube.at<uchar>(t, rho) = edgeImg.at<uchar>(y, x);
			//Scalar intensity = photo.at<uchar>(y, x);
			//cout << intensity<<"  ";
		}
	}
	cout << (int)cube.at<uchar>(31, 482) << endl;
	for (int j = 0; j < nt; j++) {
		for (int k = 1; k < nrho; k++)
			cube.at<uchar>(k, j) += cube.at<uchar>(k - 1, j);
	}
	cout << (int)cube.at<uchar>(200, 200) << endl;
	for (int j = 0; j < nt; j++) {
		for (int k = 0; k < nrho; k++) {
			int rho1 = (k + winsz) < (nrho - 1) ? (k + winsz) : (nrho - 1);
			int rho2 = (k - winsz) > 0 ? (k - winsz) : 0;
			int sum = cube.at<uchar>(rho1, j) - cube.at<uchar>(rho2, j);
			cube.at<uchar>(k, j) = sum * sum*sum;
		}
	}
	cout << (int)cube.at<uchar>(200, 200) << endl;
	for (int j = 0; j < nrho; j++) {
		for (int k = 1; k < nt; k++)
			cube.at<uchar>(j, k) += cube.at<uchar>(j, k - 1);
	}
	cout << (int)cube.at<uchar>(200, 200) << endl;
	for (int j = 0; j < nrho; j++) {
		for (int k = 0; k < nt; k++) {
			int t1 = (k + round(winsz / 2.0)) < (nt - 1) ? (k + round(winsz / 2.0)) : (nt - 1);
			int t2 = (k - round(winsz / 2.0)) > 0 ? (k - round(winsz / 2.0)) : 0;
			cube.at<uchar>(j, k) = cube.at<uchar>(j, t1) - cube.at<uchar>(j, t2);
		}
	}
	cout << (int)cube.at<uchar>(200, 200) << endl;
	Mat pcube(nrows, ncols, CV_8UC1, Scalar::all(0));
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols ; j++) {
			double t = (j - ncols / 2.0)*sin(theta) - (i - nrows / 2.0)*cos(theta) + nt / 2;
			double rho = (j - ncols / 2.0)*cos(theta) + (i - nrows / 2.0)*sin(theta) + nrho / 2;
			double divide = pow(winsz, 4)*nrows;
			pcube.at<uchar>(i, j) = saturate_cast<uchar>((cube.at<uchar>(rho, t) / divide)>=1?255:0 );
		}
	}
	cout << (int)cube.at<uchar>(200, 200) << endl;
	delete[]x;
	delete[]y;
	return pcube;
}

int main() {
	Mat src = imread("D:/vs work/data_pa.jpg");
	cout << src.channels() << endl;
	cout << (int)src.at<Vec3b>(200, 300)[0] << endl;
	cout << src.type() << endl;
	Mat edge;
	edge.create(src.size(), src.type());
	Canny(src, edge, 200, 100, 3);
	cout << (int)edge.at<uchar>(31, 482) << endl;
	cout << edge.channels() << endl;
	cout << edge.type() << endl;

	double theta = 60 / 180.0*3.1415926;
	Mat dst = localRadon(edge, theta, 20);
	imshow("radon.png", dst);
	/// 等待用户按键
	waitKey();
	return 0;
}*/


/*#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void show();//读取与显示图像
void showROI();//图像混合
void add();//ROI初级图像叠加
void canny();//显示图像边缘
void split();//分离颜色通道
void trackbar();//创建轨迹条
static void on_trackbar(int, void*);//  描述：轨迹条的回调函数 
void bright();//对比度，亮度调节函数
static void ContrastAndBright(int, void *); //亮度，对比度回调函数

Mat img;  //轨迹条函数用到的全局图像
int threshval = 160;            //轨迹条滑块对应的值，给初值160  
int g_nContrastValue; //对比度值  
int g_nBrightValue;  //亮度值  
Mat g_srcImage, g_dstImage;  //亮度，对比度调节用到的全局图像

int main()
{
	trackbar();
	waitKey(100);
	getchar();
	return 0;
}

void bright()//对比度，亮度调节函数
{
	//读入用户提供的图像  
	g_srcImage = imread("D:/vs work/luoli.png");
	if (!g_srcImage.data) { printf("Oh，no，读取g_srcImage图片错误~！\n"); }
	g_dstImage = Mat::zeros(g_srcImage.size(), g_srcImage.type());

	//设定对比度和亮度的初值  
	g_nContrastValue = 80;
	g_nBrightValue = 80;

	//创建窗口  
	namedWindow("【效果图窗口】", 1);

	//创建轨迹条  
	createTrackbar("对比度：", "【效果图窗口】", &g_nContrastValue, 300, ContrastAndBright);
	createTrackbar("亮   度：", "【效果图窗口】", &g_nBrightValue, 200, ContrastAndBright);

	//调用回调函数  
	ContrastAndBright(g_nContrastValue, 0);
	ContrastAndBright(g_nBrightValue, 0);

}
static void ContrastAndBright(int, void *) //亮度，对比度回调函数
{

	//创建窗口  
	namedWindow("【原始图窗口】", 1);

	//三个for循环，执行运算 g_dstImage(i,j) =a*g_srcImage(i,j) + b  
	for (int y = 0; y < g_srcImage.rows; y++)
	{
		for (int x = 0; x < g_srcImage.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				g_dstImage.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((g_nContrastValue*0.01)*(g_srcImage.at<Vec3b>(y, x)[c]) + g_nBrightValue);
			}
		}
	}

	//显示图像  
	imshow("【原始图窗口】", g_srcImage);
	imshow("【效果图窗口】", g_dstImage);
}

void trackbar()  //轨迹条函数
{  
	img = imread("D:/vs work/luoli.png", 0);

	//显示原图  
	namedWindow("Image", 1);
	imshow("Image", img);

	//创建处理窗口  
	namedWindow("Connected Components", 1);
	//创建轨迹条  
	createTrackbar("Threshold", "Connected Components", &threshval, 300, on_trackbar);
	on_trackbar(threshval, 0);//轨迹条回调函数 
	/*for (int i = 0; i < 5; i++) {
		cout << "......" << endl;
		cin >> threshval;
		on_trackbar(threshval, 0);//轨迹条回调函数  
		waitKey(100);
		getchar();
	}*/
/*	
}
static void on_trackbar(int, void*)//轨迹条回回调函数
{
	Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);

	//定义点和向量  
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//查找轮廓  
	findContours(bw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//初始化dst  
	Mat dst = Mat::zeros(img.size(), CV_8UC3);
	//开始处理  
	if (!contours.empty() && !hierarchy.empty())
	{
		//遍历所有顶层轮廓，随机生成颜色值绘制给各连接组成部分  
		int idx = 0;
		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			Scalar color((rand() & 255), (rand() & 255), (rand() & 255));
			//绘制填充轮廓  
			drawContours(dst, contours, idx, color, CV_FILLED, 8, hierarchy);
		}
	}
	//显示窗口  
	imshow("Connected Components", dst);
	//imwrite("D:/vs work/luolitrack.png", dst);
}


//分离颜色通道
void split()
{	
	Mat srcImage;
	Mat imageROI;
	Mat logoImage;
	vector<Mat> channels;
	srcImage = imread("D:/vs work/data_pa.jpg");
	logoImage = imread("D:/vs work/data_logo.jpg",0);
	// 把一个3通道图像转换成3个单通道图像  
	split(srcImage, channels);//分离色彩通道  
	imageROI = channels.at(0);
	addWeighted(imageROI(Rect(700, 250, logoImage.cols, logoImage.rows)),
		1.0,logoImage,  0.5, 0.0, imageROI(Rect(700, 250, logoImage.cols, logoImage.rows)));
	merge(channels, srcImage);

	namedWindow("sample");
	imshow("sample", srcImage);
}

//图像的权重混合
void showROI()
{
	double alphaValue = 0.5;
	double betaValue;

	Mat img1, img2, img3;

	img1 = imread("D:/vs work/grass.jpg");
	img2 = imread("D:/vs work/rain.jpg");

	betaValue = 1.0 - alphaValue;
	addWeighted(img1, alphaValue, img2, betaValue, 0.0, img3);

	namedWindow("test1");
	imshow("test1", img3);
	imwrite("D:/vs work/grass+rain.jpg",img3);
}


//选择区域进行区域图像的叠加，对图像进行初级的混合
void add()
{
	Mat img = imread("D:/vs work/data_pa.jpg");
	Mat img_logo = imread("D:/vs work/data_logo.jpg");

	Mat imgROI = img(Rect(200, 250,img_logo.cols,img_logo.rows ));

	Mat mask = imread("D:/vs work/data_logo.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	img_logo.copyTo(imgROI,mask);//这里加mask是为了让欧共图片下没有覆盖的图像出现
	namedWindow("test1");
	imshow("test1", img);
	imwrite("D:/vs work/data_quan.jpg",img);
	
}



//canny算子显示图像边缘
void canny()
{
	Mat img = imread("D:/vs work/1.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	
	namedWindow("Test",WINDOW_NORMAL);
	if (img.empty())
	{
		cout << "图片指针为空，请输入正确的图片路径";
	}

	imshow("Test1", img);

	//进行canny算子操作并且返回计算结果
	Mat cannyResult;
	Canny(img, cannyResult, 50, 150);
	//imshow("Test", cannyResult);
	imshow("Test", cannyResult);
	imwrite("D:/vs work/canny1.jpg", cannyResult);
	
}

//读取与显示图像
void show()
{
	Mat image = imread("D:\\vs work\\1.jpg");  //存放自己图像的路径 
	imshow("显示图像", image);
}*/