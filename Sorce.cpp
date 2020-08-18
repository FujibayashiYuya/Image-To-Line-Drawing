#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<stdio.h>
#include<math.h>

using namespace cv;
using namespace std;

int main(){
  Mat img, gray, dila, diff;
  //画像読み込み + 例外処理
  img = imread("images/src.jpg");
  if(img.empty() == true){
    cout << "画像が存在しません" << endl;
    return 0;
  }
  //画像をグレースケール化
  cvtColor(img, gray, COLOR_BGR2GRAY, 1);
  //黒い箇所を狭める
  dilate(gray, dila, Mat(), Point(-1, -1), 1);
  absdiff(gray, dila, diff);
  
  namedWindow("Risult_Image");
	imshow("result_Image", diff);
	imwrite("result.jpg", diff);
	waitKey(0);
}