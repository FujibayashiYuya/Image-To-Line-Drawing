#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<stdio.h>
#include<math.h>

using namespace cv;
using namespace std;

Mat lookuptable(Mat img, int min, int max) {
	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.data;
	int diff = max - min;
	for (int i = 0; i < 256; i++) {
		p[i] = min + i * diff / 255;
	}
	Mat dst;
	LUT(img, lookUpTable, dst);
	return dst;
}

//Kmeans法を行う関数
Mat Kmeans(Mat img) {
	int clus = 6;
	Mat points;
	img.convertTo(points, CV_32FC3);
	points = points.reshape(3, img.rows*img.cols);

	Mat_<int> clusters(points.size(), CV_32SC1);
	Mat centers;//クラスタ中心
	kmeans(points, clus, clusters, TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10, 1.0), 1, cv::KMEANS_PP_CENTERS, centers); //k-meansを適用
	Mat dst(img.size(), img.type());
	//始点,終点イテレータ
	cv::MatIterator_<cv::Vec3b> itd = dst.begin<cv::Vec3b>();
	cv::MatIterator_<cv::Vec3b> itd_end = dst.end<cv::Vec3b>();
	for (int i = 0; itd != itd_end; ++itd, ++i) {

		cv::Vec3f &color = centers.at<cv::Vec3f>(clusters(i), 0);
		(*itd)[0] = cv::saturate_cast<uchar>(color[0]);
		(*itd)[1] = cv::saturate_cast<uchar>(color[1]);
		(*itd)[2] = cv::saturate_cast<uchar>(color[2]);
	}
	return dst;
}

//グレースケール→白強調→差分→反転（線画抽出）
Mat Median1(Mat img) {
	Mat gray, blu, dila, diff, edge;
	medianBlur(img, blu, 3);
	//画像をグレースケール化
	cvtColor(blu, gray, COLOR_BGR2GRAY, 1);
	//黒い箇所を狭める
	dilate(gray, dila, Mat(), Point(-1, -1), 1);
	absdiff(gray, dila, diff);
	//bitwise_not(diff, edge);
	return diff;
}

int main() {
	Mat img, result, canny, median, km, test;
	//画像読み込み + 例外処理
	img = imread("images/tree2.jpg");
	if (img.empty() == true) {
		cout << "画像が存在しません" << endl;
		return 0;
	}
	//blur(img, blu, Size(3, 3), Point(-1,-1), BORDER_DEFAULT);
	Canny(img, canny, 125, 255);
	km = Kmeans(img);
	medianBlur(km, km, 3);
	km = Median1(km);
	median = Median1(img);
	bitwise_not(median, test);

	imwrite("km.jpg", km);
	imwrite("median.jpg", median);
	imwrite("canny.jpg", canny);
	imwrite("metest.jpg", test);

	bitwise_and(km, canny, result);
	bitwise_not(result, result);

	//画像を表示・保存
	namedWindow("Risult_Image");
	imshow("result_Image", result);
	imwrite("result.jpg", result);
	waitKey(0);
}
//http://azp2.sakuraweb.com/photo/rinkaku/index.html
//LUTをc++ Opencvで https://cvtech.cc/lut/
//Kmeans https://tora-k.com/2019/11/28/anime_filter/
//http://gahag.net/000312-tree-landscape/ 写真素材
//Opencv(Python)で画像を淡く　https://www.blog.umentu.work/python-opencv3%E3%81%A7%E3%82%B3%E3%83%B3%E3%83%88%E3%83%A9%E3%82%B9%E3%83%88%E3%82%92%E4%BD%8E%E6%B8%9B%E8%96%84%E3%81%8F%E3%81%99%E3%82%8B/
