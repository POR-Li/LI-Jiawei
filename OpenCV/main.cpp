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
				y[q] = i;//��
				x[q] = j;//��
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
	/// �ȴ��û�����
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
				y[q] = i;//��
				x[q] = j;//��
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
	/// �ȴ��û�����
	waitKey();
	return 0;
}*/


/*#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void show();//��ȡ����ʾͼ��
void showROI();//ͼ����
void add();//ROI����ͼ�����
void canny();//��ʾͼ���Ե
void split();//������ɫͨ��
void trackbar();//�����켣��
static void on_trackbar(int, void*);//  �������켣���Ļص����� 
void bright();//�Աȶȣ����ȵ��ں���
static void ContrastAndBright(int, void *); //���ȣ��ԱȶȻص�����

Mat img;  //�켣�������õ���ȫ��ͼ��
int threshval = 160;            //�켣�������Ӧ��ֵ������ֵ160  
int g_nContrastValue; //�Աȶ�ֵ  
int g_nBrightValue;  //����ֵ  
Mat g_srcImage, g_dstImage;  //���ȣ��Աȶȵ����õ���ȫ��ͼ��

int main()
{
	trackbar();
	waitKey(100);
	getchar();
	return 0;
}

void bright()//�Աȶȣ����ȵ��ں���
{
	//�����û��ṩ��ͼ��  
	g_srcImage = imread("D:/vs work/luoli.png");
	if (!g_srcImage.data) { printf("Oh��no����ȡg_srcImageͼƬ����~��\n"); }
	g_dstImage = Mat::zeros(g_srcImage.size(), g_srcImage.type());

	//�趨�ԱȶȺ����ȵĳ�ֵ  
	g_nContrastValue = 80;
	g_nBrightValue = 80;

	//��������  
	namedWindow("��Ч��ͼ���ڡ�", 1);

	//�����켣��  
	createTrackbar("�Աȶȣ�", "��Ч��ͼ���ڡ�", &g_nContrastValue, 300, ContrastAndBright);
	createTrackbar("��   �ȣ�", "��Ч��ͼ���ڡ�", &g_nBrightValue, 200, ContrastAndBright);

	//���ûص�����  
	ContrastAndBright(g_nContrastValue, 0);
	ContrastAndBright(g_nBrightValue, 0);

}
static void ContrastAndBright(int, void *) //���ȣ��ԱȶȻص�����
{

	//��������  
	namedWindow("��ԭʼͼ���ڡ�", 1);

	//����forѭ����ִ������ g_dstImage(i,j) =a*g_srcImage(i,j) + b  
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

	//��ʾͼ��  
	imshow("��ԭʼͼ���ڡ�", g_srcImage);
	imshow("��Ч��ͼ���ڡ�", g_dstImage);
}

void trackbar()  //�켣������
{  
	img = imread("D:/vs work/luoli.png", 0);

	//��ʾԭͼ  
	namedWindow("Image", 1);
	imshow("Image", img);

	//����������  
	namedWindow("Connected Components", 1);
	//�����켣��  
	createTrackbar("Threshold", "Connected Components", &threshval, 300, on_trackbar);
	on_trackbar(threshval, 0);//�켣���ص����� 
	/*for (int i = 0; i < 5; i++) {
		cout << "......" << endl;
		cin >> threshval;
		on_trackbar(threshval, 0);//�켣���ص�����  
		waitKey(100);
		getchar();
	}*/
/*	
}
static void on_trackbar(int, void*)//�켣���ػص�����
{
	Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);

	//����������  
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//��������  
	findContours(bw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//��ʼ��dst  
	Mat dst = Mat::zeros(img.size(), CV_8UC3);
	//��ʼ����  
	if (!contours.empty() && !hierarchy.empty())
	{
		//�������ж������������������ɫֵ���Ƹ���������ɲ���  
		int idx = 0;
		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			Scalar color((rand() & 255), (rand() & 255), (rand() & 255));
			//�����������  
			drawContours(dst, contours, idx, color, CV_FILLED, 8, hierarchy);
		}
	}
	//��ʾ����  
	imshow("Connected Components", dst);
	//imwrite("D:/vs work/luolitrack.png", dst);
}


//������ɫͨ��
void split()
{	
	Mat srcImage;
	Mat imageROI;
	Mat logoImage;
	vector<Mat> channels;
	srcImage = imread("D:/vs work/data_pa.jpg");
	logoImage = imread("D:/vs work/data_logo.jpg",0);
	// ��һ��3ͨ��ͼ��ת����3����ͨ��ͼ��  
	split(srcImage, channels);//����ɫ��ͨ��  
	imageROI = channels.at(0);
	addWeighted(imageROI(Rect(700, 250, logoImage.cols, logoImage.rows)),
		1.0,logoImage,  0.5, 0.0, imageROI(Rect(700, 250, logoImage.cols, logoImage.rows)));
	merge(channels, srcImage);

	namedWindow("sample");
	imshow("sample", srcImage);
}

//ͼ���Ȩ�ػ��
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


//ѡ�������������ͼ��ĵ��ӣ���ͼ����г����Ļ��
void add()
{
	Mat img = imread("D:/vs work/data_pa.jpg");
	Mat img_logo = imread("D:/vs work/data_logo.jpg");

	Mat imgROI = img(Rect(200, 250,img_logo.cols,img_logo.rows ));

	Mat mask = imread("D:/vs work/data_logo.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	img_logo.copyTo(imgROI,mask);//�����mask��Ϊ����ŷ��ͼƬ��û�и��ǵ�ͼ�����
	namedWindow("test1");
	imshow("test1", img);
	imwrite("D:/vs work/data_quan.jpg",img);
	
}



//canny������ʾͼ���Ե
void canny()
{
	Mat img = imread("D:/vs work/1.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	
	namedWindow("Test",WINDOW_NORMAL);
	if (img.empty())
	{
		cout << "ͼƬָ��Ϊ�գ���������ȷ��ͼƬ·��";
	}

	imshow("Test1", img);

	//����canny���Ӳ������ҷ��ؼ�����
	Mat cannyResult;
	Canny(img, cannyResult, 50, 150);
	//imshow("Test", cannyResult);
	imshow("Test", cannyResult);
	imwrite("D:/vs work/canny1.jpg", cannyResult);
	
}

//��ȡ����ʾͼ��
void show()
{
	Mat image = imread("D:\\vs work\\1.jpg");  //����Լ�ͼ���·�� 
	imshow("��ʾͼ��", image);
}*/