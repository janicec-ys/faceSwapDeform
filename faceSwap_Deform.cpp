// expressionDetect.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include "source.cpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>



// number of training images
#define M 6
#define M_PI 3.14159265358979323846

using namespace std;
using namespace dlib;

// Calculate Delaunay triangles for set of points
// Returns the vector of indices of 3 points for each triangle
static void calculateDelaunayTriangles(cv::Rect rect, std::vector<cv::Point2f> &points, std::vector< std::vector<int> > &delaunayTri) {

	// Create an instance of Subdiv2D
	cv::Subdiv2D subdiv(rect);

	// Insert points into subdiv
	for (std::vector<cv::Point2f>::iterator it = points.begin(); it != points.end(); it++)
	{
		cv::Point2f p = *it;
		if (0 <= p.x && p.x <= rect.width && 0 <= p.y && p.y < rect.height)
			subdiv.insert(*it);
	}
	std::vector<cv::Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	std::vector<cv::Point2f> pt(3);
	std::vector<int> ind(3);

	for (size_t i = 0; i < triangleList.size(); i++)
	{
		cv::Vec6f t = triangleList[i];
		pt[0] = cv::Point2f(t[0], t[1]);
		pt[1] = cv::Point2f(t[2], t[3]);
		pt[2] = cv::Point2f(t[4], t[5]);

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
			for (int j = 0; j < 3; j++)
				for (size_t k = 0; k < points.size(); k++)
					if (abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
						ind[j] = k;

			delaunayTri.push_back(ind);
		}
	}

}


// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(cv::Mat &warpImage, cv::Mat &src, std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri)
{
	// Given a pair of triangles, find the affine transform.
	cv::Mat warpMat = getAffineTransform(srcTri, dstTri);

	// Apply the Affine Transform just found to the src image
	warpAffine(src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
}




// Warps and alpha blends triangular regions from img1 and img2 to img
// t12: triangles mapped from img2 to img1, t22: triangles of img2
void warpTriangle(cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2f> &t1, std::vector<cv::Point2f> &t2, cv::Mat &blank)
{

	cv::Rect r1 = boundingRect(t1);
	cv::Rect r2 = boundingRect(t2);

	// Offset points by left top corner of the respective rectangles
	std::vector<cv::Point2f> t1Rect, t2Rect;
	std::vector<cv::Point> t2RectInt;
	for (int i = 0; i < 3; i++)
	{

		t1Rect.push_back(cv::Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(cv::Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
		t2RectInt.push_back(cv::Point(t2[i].x - r2.x, t2[i].y - r2.y)); // for fillConvexPoly

	}

	// Get mask by filling triangle
	cv::Mat mask = cv::Mat::zeros(r2.height, r2.width, CV_32FC3);
	fillConvexPoly(mask, t2RectInt, cv::Scalar(1.0, 1.0, 1.0), 16, 0);

	// Apply warpImage to small rectangular patches
	cv::Mat img1Rect;
	img1(r1).copyTo(img1Rect);

	cv::Mat img2Rect = cv::Mat::zeros(r2.height, r2.width, img1Rect.type());

	applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);

	multiply(img2Rect, mask, img2Rect);
	multiply(img2(r2), cv::Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
	multiply(blank(r2), cv::Scalar(1.0, 1.0, 1.0) - mask, blank(r2));
	/*img2(r2) = img2(r2) + img2Rect;*/
	blank(r2) = blank(r2) + img2Rect;



}



static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r)
{

	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right(), r.bottom()));

}

bool BilinearInterpolation(cv::Mat frame, double x, double y, double rgb[3])
{
	cv::Vec3b Q11, Q12, Q21, Q22;
	int x1, x2, y1, y2;
	x1 = floor(x);
	x2 = ceil(x);
	y1 = floor(y);
	y2 = ceil(y);

	if (x1 == x2)
	{
		if (y1 == y2) // x and y are integers
		{
			Q11 = frame.at<cv::Vec3b>(y1, x1);
			rgb[0] = Q11[0];
			rgb[1] = Q11[1];
			rgb[2] = Q11[2];
			return true;
		}
		else // x is an integer, y is not.
		{
			Q11 = frame.at<cv::Vec3b>(y1, x1);
			Q12 = frame.at<cv::Vec3b>(y2, x1);
			rgb[0] = (Q11[0] * (y2 - y) + Q12[0] * (y - y1)) / (y2 - y1);
			rgb[1] = (Q11[1] * (y2 - y) + Q12[1] * (y - y1)) / (y2 - y1);
			rgb[2] = (Q11[2] * (y2 - y) + Q12[2] * (y - y1)) / (y2 - y1);
			return true;
		}
	}
	// y is an integer, x is not.
	if (y1 == y2)
	{
		Q11 = frame.at<cv::Vec3b>(y1, x1);
		Q21 = frame.at<cv::Vec3b>(y1, x2);
		rgb[0] = (Q11[0] * (x2 - x) + Q21[0] * (x - x1)) / (x2 - x1);
		rgb[1] = (Q11[1] * (x2 - x) + Q21[1] * (x - x1)) / (x2 - x1);
		rgb[2] = (Q11[2] * (x2 - x) + Q21[2] * (x - x1)) / (x2 - x1);
		return true;
	}

	// When x and y are not intergers
	Q11 = frame.at<cv::Vec3b>(y1, x1);
	Q12 = frame.at<cv::Vec3b>(y2, x1);
	Q21 = frame.at<cv::Vec3b>(y1, x2);
	Q22 = frame.at<cv::Vec3b>(y2, x2);

	// Compute the RGB at (x, y)
	rgb[0] = (Q11[0] * (x2 - x)*(y2 - y) + Q21[0] * (x - x1)*(y2 - y) + Q12[0] * (x2 - x)*(y - y1) + Q22[0] * (x - x1)*(y - y1)) / ((x2 - x1)*(y2 - y1));
	rgb[1] = (Q11[1] * (x2 - x)*(y2 - y) + Q21[1] * (x - x1)*(y2 - y) + Q12[1] * (x2 - x)*(y - y1) + Q22[1] * (x - x1)*(y - y1)) / ((x2 - x1)*(y2 - y1));
	rgb[2] = (Q11[2] * (x2 - x)*(y2 - y) + Q21[2] * (x - x1)*(y2 - y) + Q12[2] * (x2 - x)*(y - y1) + Q22[2] * (x - x1)*(y - y1)) / ((x2 - x1)*(y2 - y1));

	return true;
}


static int N = 0;
int mode = 1; // mode 1 for Swaping; mode 2 for expression

int main(int argc, char * argv[])
{
	//Loads happy and sad images
	cv::Mat img_happy = cv::imread("D:\\Projects\\CV_project\\opcv_test\\face_landmark\\data\\happy.jpg", 1);
	cv::Mat img_sad = cv::imread("D:\\Projects\\CV_project\\opcv_test\\face_landmark\\data\\sad.jpg", 1);

	//Reads the image size 
	cv::Size imgSz = cv::imread("D:\\Projects\\CV_project\\opcv_test\\face_landmark\\data\\happy00.png", 0).size();
	//Calculate the vector length for one image
	N = imgSz.width * imgSz.height;
	//cv::Mat projctedTrain;
	//cv::Mat projectionMatrix;
	//PCA(N, projctedTrain, projectionMatrix);
	//Training images
	cv::Mat S = cv::Mat::zeros(N, 1, CV_32FC1);
	cv::Mat I = cv::Mat(N, M, CV_32FC1);
	char file[64];
	for (int i = 0; i < M / 2; i++)
	{
		sprintf(file, "D:\\Projects\\CV_project\\opcv_test\\face_landmark\\data\\happy%02d.png", i);
		cv::Mat m = cv::imread(file, CV_LOAD_IMAGE_GRAYSCALE);
		m = m.t();
		m = m.reshape(1, N);
		m.convertTo(m, CV_32FC1);
		m.copyTo(I.col(i));
		S = S + m;
	}
	for (int i = 0; i < M / 2; i++)
	{
		sprintf(file, "D:\\Projects\\CV_project\\opcv_test\\face_landmark\\data\\sad%02d.png", i);
		cv::Mat m = cv::imread(file, CV_LOAD_IMAGE_GRAYSCALE);
		m = m.t();
		m = m.reshape(1, N);
		m.convertTo(m, CV_32FC1);
		m.copyTo(I.col(i + M / 2));
		S = S + m;
	}

	//calculate eigenvectors
	cv::Mat mean = S / (float)M;
	cv::Mat A = cv::Mat(N, M, CV_32FC1);
	for (int i = 0; i < M; i++)
	{
		A.col(i) = I.col(i) - mean;
	}
	cv::Mat C = A.t() * A;
	cv::Mat EVal, EVec;
	cv::eigen(C, EVal, EVec);

	//compute projection matrix 
	cv::Mat U = A * EVec;
	cv::Mat projectionMatrix = U.t();

	// project the training set to the faces space
	cv::Mat projctedTrain = projectionMatrix * A;



	//start camera
	cv::VideoCapture vc;
	char ch;
	cv::Mat frame;
	//cv::Mat frame;
	bool isVideo = true;

	//setup dlib
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;

	int happy_count = 0;
	int sad_count = 0;

	try {
		vc.open(0);//open default camera
		deserialize("D:\\Projects\\CV_project\\opcv_test\\face_landmark\\shape_predictor_68_face_landmarks.dat") >> pose_model;
	}
	catch (cv::Exception e) {
		frame = cv::Mat::eye(100, 100, CV_32FC1);
		isVideo = false;
	}
	catch (serialization_error& e)
	{
		cout << "you need dlib's default face landmarking model file to run this example." << endl;
		cout << "you can get it from the following url: " << endl;
		cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}


	while (1)
	{
		if (isVideo) {
			try
			{
				vc >> frame;
				//vc >> frame;
				cv_image<bgr_pixel> cimg(frame);

				cv::Mat result;
				// detect faces
				std::vector<rectangle> faces = detector(cimg);
				// get the biggest face
				rectangle maxFace;
				for (int i = 0; i < faces.size(); i++)
				{
					if (faces[i].area() > maxFace.area()) maxFace = faces[i];
				}

				rectangle twoFaces[2];
				// swap faces
				if (mode == 1)
				{
					if (faces.size() >= 2)
					{
						// Choose the biggest two faces
						if (faces.size() > 2)
						{
							//rectangle maxRect, secRect;
							for (int i = 0; i < faces.size(); i++)
							{
								if (faces[i].area() > twoFaces[0].area()) twoFaces[0] = faces[i];
								if (twoFaces[1].area() < faces[i].area() && faces[i].area() < twoFaces[0].area()) twoFaces[1] = faces[i];
							}
						}
						else
						{
							twoFaces[0] = faces[0];
							twoFaces[1] = faces[1];
						}

						// make sure every landmark points are within frame
						int swapFlag = 1;
						for (int i = 0; i < 2; i++)
						{
							boolean bxL = 0 <= twoFaces[i].left() && twoFaces[i].left() <= frame.cols;
							boolean bxR = 0 <= twoFaces[i].right() && twoFaces[i].right() <= frame.cols;
							boolean byT = 0 <= twoFaces[i].top() && twoFaces[i].top() <= frame.rows;
							boolean byB = 0 <= twoFaces[i].bottom() && twoFaces[i].bottom() <= frame.rows;
							full_object_detection shape = pose_model(cimg, twoFaces[i]);
							if ((bxL && bxR && byT && byB) == false)
							{
								swapFlag = 0;
								break;
							}
							else
							{
								for (int j = 5; j <= 10; j++)
								{
									int posx = shape.part(j).x();
									int posy = shape.part(j).y();
									if (!(0 <= posx && posx <= frame.cols && 0 <= posy && posy <= frame.rows))
									{
										swapFlag = 0;
										break;
									}
								}
							}
						}

						if (swapFlag == 1)
						{

							cv::Mat swapTemp;
							frame.copyTo(swapTemp);
							// swapTemp1 has two holes for two faces
							cv::Mat swapTemp1;

							//convert
							swapTemp.convertTo(swapTemp, CV_32F);
							swapTemp.copyTo(swapTemp1);

							full_object_detection shape1, shape2;
							std::vector<cv::Point2f> points1, points2;
							for (int i = 0; i < 68; i++)
							{
								shape1 = pose_model(cimg, twoFaces[0]);
								shape2 = pose_model(cimg, twoFaces[1]);
								points1.push_back(cv::Point2f(shape1.part(i).x(), shape1.part(i).y()));
								points2.push_back(cv::Point2f(shape2.part(i).x(), shape2.part(i).y()));
							}


							// find convex hull
							std::vector<cv::Point2f> hull11, hull21;
							std::vector<cv::Point2f> hull12, hull22;
							std::vector<int> hullIndex1, hullIndex2;
							cv::convexHull(points2, hullIndex2, false, false);
							cv::convexHull(points1, hullIndex1, false, false);




							for (int i = 0; i < hullIndex2.size(); i++)
							{

								hull12.push_back(points1[hullIndex2[i]]);
								hull22.push_back(points2[hullIndex2[i]]);


							}
							for (int i = 0; i < hullIndex1.size(); i++)
							{
								hull11.push_back(points1[hullIndex1[i]]);
								hull21.push_back(points2[hullIndex1[i]]);
							}

							// find delaunary triangulation for points on the convex hull
							std::vector< std::vector<int> > dt1, dt2;
							cv::Rect rect(0, 0, swapTemp.cols, swapTemp.rows);


							calculateDelaunayTriangles(rect, hull22, dt2);
							calculateDelaunayTriangles(rect, hull11, dt1);

							//apply affine transformation to Delaunay triangles
							cv::Mat face1Warped, face2Warped;
							swapTemp.copyTo(face1Warped);
							swapTemp.copyTo(face2Warped);
							face1Warped = cv::Scalar(0);
							face2Warped = cv::Scalar(0);

							for (size_t i = 0; i < dt2.size(); i++)
							{
								std::vector<cv::Point2f> t1, t2;
								//Get points for img1, img2 corresponding to the triangles
								for (size_t j = 0; j < 3; j++)
								{
									t1.push_back(hull12[dt2[i][j]]);
									t2.push_back(hull22[dt2[i][j]]);
								}
								warpTriangle(swapTemp, swapTemp1, t1, t2, face1Warped);
							}


							for (size_t i = 0; i < dt1.size(); i++)
							{
								std::vector<cv::Point2f> t3, t4;
								//Get points for img1, img2 corresponding to the triangles
								for (size_t j = 0; j < 3; j++)
								{
									t3.push_back(hull21[dt1[i][j]]);
									t4.push_back(hull11[dt1[i][j]]);
								}
								warpTriangle(swapTemp, swapTemp1, t3, t4, face2Warped);
							}


							cv::Mat swapped;
							swapTemp1.copyTo(swapped);
							swapped = swapTemp1 + face1Warped + face2Warped;



							//Calculate mask
							std::vector<cv::Point> hull8U2, hull8U1;
							for (int i = 0; i < hull22.size(); i++)
							{
								cv::Point pt(hull22[i].x, hull22[i].y);
								hull8U2.push_back(pt);
							}

							for (int i = 0; i < hull11.size(); i++)
							{
								cv::Point pt(hull11[i].x, hull11[i].y);
								hull8U1.push_back(pt);
							}



							//for seamless
							cv::Mat mask1 = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
							cv::Mat mask2 = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
							fillConvexPoly(mask1, &hull8U1[0], hull8U1.size(), cv::Scalar(255, 255, 255));
							fillConvexPoly(mask2, &hull8U2[0], hull8U2.size(), cv::Scalar(255, 255, 255));


							////clone seamlessly
							cv::Rect r2 = cv::boundingRect(hull22);
							cv::Rect r1 = cv::boundingRect(hull11);
							cv::Point center2 = (r2.tl() + r2.br()) / 2;
							cv::Point center1 = (r1.tl() + r1.br()) / 2;

							cv::Mat output1, output2;

							//
							swapped.convertTo(swapped, CV_8UC3);
							frame.convertTo(frame, CV_8UC3);

							cv::seamlessClone(swapped, frame, mask1, center1, output1, cv::NORMAL_CLONE);
							cv::seamlessClone(swapped, output1, mask2, center2, output2, cv::NORMAL_CLONE);

							//cv::Mat result;
							output2.copyTo(result);
							//cv::imshow("deform", output2);

						}
					} 
					else {
						frame.copyTo(result);
					}
				}


				// convert dlib rectangle to opencv rect
				cv::Rect *roi = 0;
				roi = &dlibRectangleToOpenCV(maxFace);

				// get landmarks on mouth (#46~#67)
				std::vector<cv::Point2f> mouthLM;
				for (int i = 48; i < 68; i++)
				{
					mouthLM.push_back(cv::Point2f(pose_model(cimg, maxFace).part(i).x(),
						pose_model(cimg, maxFace).part(i).y()));
				}

				cv::Rect rectMouth = cv::boundingRect(mouthLM);

				//for testing
				/*cv::rectangle(frame, rectMouth.tl(), rectMouth.br(), (255, 255, 255), 3);
				cv::imshow("face", frame);*/

				int suby = roi->height * 0.5;
				roi->height -= suby;
				roi->y += suby;
				int subx = (roi->width - roi->height) / 2 * 0.7;
				roi->width -= subx * 2;
				roi->x += subx;
				cv::Mat imgROI = frame(*roi);
				cv::Mat *subimg = &cv::Mat(100, 100 * 0.7, CV_8UC3);
				cv::Mat *subimg_gray = &cv::Mat(100, 100 * 0.7, CV_8UC1);
				cv::Size size(100, 100 * 0.7);
				//cv::resize(frame, *subimg, size);
				cv::resize(imgROI, *subimg, size);
				cv::cvtColor(*subimg, *subimg_gray, cv::COLOR_BGR2GRAY);
				cv::equalizeHist(*subimg_gray, *subimg_gray);
				// getting training data
				/*char ck = (char)cv::waitKey(10);
				switch (ck)
				{
				char file[32];
				case 'h':
				sprintf(file, "D:\\Projects\\CV_project\\opcv_test\\face_landmark\\data\\happy%02d.png", happy_count);
				cv::imwrite(file, *subimg_gray);
				happy_count++;
				cv::waitKey(1000);
				break;
				case 'a':
				sprintf(file, "D:\\Projects\\CV_project\\opcv_test\\face_landmark\\data\\sad%02d.png", sad_count);
				cv::imwrite(file, *subimg_gray);
				sad_count++;
				cv::waitKey(1000);
				break;
				}*/

				// recognize mood
				double min = 0.0;
				int min_ind;
				cv::Mat subimgGray;
				subimg_gray->copyTo(subimgGray);
				subimgGray = subimgGray.t();
				subimgGray = subimgGray.reshape(1, N);
				subimgGray.convertTo(subimgGray, CV_32FC1);
				cv::Mat proj = projectionMatrix * subimgGray;
				// find the minimum distance vector
				for (int i = 0; i < M; i++)
				{
					double n = norm(proj - projctedTrain.col(i));
					if (min > n || i == 0)
					{
						min = n;
						min_ind = i;
					}
				}
				/*if (min_ind > M/2) cv::imshow("logo", img_sad);
				else cv::imshow("logo", img_happy);
				cv::moveWindow("logo", 670, 200);*/

				/*cv::imshow("face", frame);
				cv::moveWindow("face", 0, 0);

				cv::imshow("gray", *subimg_gray);
				cv::moveWindow("gray", 670, 0);*/

				// happy = 0, sad = 1
				int mood = 0;
				if (min_ind > M / 2) mood = 1;


				cv::Mat curMouth, curFrame, output;
				frame.copyTo(output);
				if (mode == 2) {
					
					if (mood == 0)
					{
						int w_frame = frame.cols;
						int h_frame = frame.rows;
						int w = rectMouth.width; // Change this to mouth max width
						int h = rectMouth.height; // Change this to mouth max height
						int x0 = rectMouth.x; // Change this to left most x of mouth contour 
						int y0 = rectMouth.y; // Change this to up most y of mouth contour 
												// Giving the bounding box
						int r0 = y0 - h / 2;
						int rt = y0 + 3 * h / 2;
						int c0 = x0;
						int ct = x0 + w;
						int cm = (c0 + ct) / 2;
						int sc = (rt - r0) / 4;

						cv::Mat curFace(2 * h, w, CV_8UC3);
						for (int r = 0; r < rt - r0; r++) {
							for (int c = 0; c < ct - c0; c++) {
								curFace.at<cv::Vec3b>(r, c)[0] = 0;
								curFace.at<cv::Vec3b>(r, c)[1] = 255;
								curFace.at<cv::Vec3b>(r, c)[2] = 0;

								double x2 = 0.0;
								double y2 = 0.0;

								//Project((double)c, (double)r, x2, y2, homInv);
								x2 = c + c0;
								y2 = r + r0 + sc*cos(double((x2 - cm)) / (ct - cm)*M_PI);
								double rgb2[3];
								if (x2 < w_frame && y2 < h_frame && x2 >= 0 && y2 >= 0) {
									BilinearInterpolation(frame, x2, y2, rgb2);
									if (rgb2[0] != 0 || rgb2[1] != 0 || rgb2[2] != 0) {
										curFace.at<cv::Vec3b>(r, c)[0] = rgb2[0];
										curFace.at<cv::Vec3b>(r, c)[1] = rgb2[1];
										curFace.at<cv::Vec3b>(r, c)[2] = rgb2[2];
									}
								}

							}
						}

						curFace.copyTo(curMouth);
						curMouth.convertTo(curMouth, CV_32F);

						frame.copyTo(curFrame);
						curFrame.convertTo(curFrame, CV_32F);

						multiply(curFrame(rectMouth), cv::Scalar(0.0, 0.0, 0.0), curFrame(rectMouth));

						cv::Mat curMouth_sub = cv::Mat(rectMouth.size(), CV_32F);
						cv::resize(curMouth, curMouth_sub, rectMouth.size());
						curFrame(rectMouth) = curFrame(rectMouth) + curMouth_sub;


						std::vector<cv::Point> hull8U;
						for (int i = 0; i < rectMouth.width; i++)
						{
							for (int j = 0; j < rectMouth.height; j++)
							{
								hull8U.push_back(cv::Point(rectMouth.x + i, rectMouth.y + j));
							}
						}

						cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
						fillConvexPoly(mask, &hull8U[0], hull8U.size(), cv::Scalar(255, 255, 255));

						cv::Point center = (rectMouth.tl() + rectMouth.br()) / 2;

						curFrame.convertTo(curFrame, CV_8UC3);
						frame.convertTo(frame, CV_8UC3);
						cv::seamlessClone(curFrame, frame, mask, center, result, cv::NORMAL_CLONE);

						//curFrame.convertTo(curFrame, CV_8UC3);
						//mask.copyTo(result);
						cv::imshow("mask",mask);

					}
					else {
						frame.copyTo(result);
					}
				}


				cv::imshow("deform", result);
				cv::moveWindow("deform", 0, 0);
				if (mode == 2)
				{
					if (mood == 1) cv::imshow("logo", img_sad);
					else cv::imshow("logo", img_happy);
					cv::moveWindow("logo", 670, 200);
				} 

			}
			catch (const std::exception& e)
			{
				cout << e.what() << endl;
			}
		}

		char c = (char)cv::waitKey(10);
		if (c == 27)
			break;
		switch (c)
		{
		case 's':
			mode = 1;
			break;
		case 'e':
			mode = 2;
			break;

		default:
			;
		}
	}
	if (isVideo)
		vc.release();
	return 0;
}












//// expressionDetect.cpp : Defines the entry point for the console application.
////
//
//
//#include "stdafx.h"
//#include "source.cpp"
//#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing/render_face_detections.h>
//#include <dlib/image_processing.h>
//#include <dlib/gui_widgets.h>
//#include <dlib/image_io.h>
//#include <dlib/opencv.h>
//#include <iostream>
//#include "opencv2/objdetect.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//
//
//// number of training images
//#define M 8
//#define M_PI 3.14159265358979323846
//
//using namespace std;
//using namespace dlib;
//
//bool BilinearInterpolation(cv::Mat frame, double x, double y, double rgb[3])
//{
//	cv::Vec3b Q11, Q12, Q21, Q22;
//	int x1, x2, y1, y2;
//	x1 = floor(x);
//	x2 = ceil(x);
//	y1 = floor(y);
//	y2 = ceil(y);
//
//	if (x1 == x2)
//	{
//		if (y1 == y2) // x and y are integers
//		{
//			Q11 = frame.at<cv::Vec3b>(y1, x1);
//			rgb[0] = Q11[0];
//			rgb[1] = Q11[1];
//			rgb[2] = Q11[2];
//			return true;
//		}
//		else // x is an integer, y is not.
//		{
//			Q11 = frame.at<cv::Vec3b>(y1, x1);
//			Q12 = frame.at<cv::Vec3b>(y2, x1);
//			rgb[0] = (Q11[0] * (y2 - y) + Q12[0] * (y - y1)) / (y2 - y1);
//			rgb[1] = (Q11[1] * (y2 - y) + Q12[1] * (y - y1)) / (y2 - y1);
//			rgb[2] = (Q11[2] * (y2 - y) + Q12[2] * (y - y1)) / (y2 - y1);
//			return true;
//		}
//	}
//	// y is an integer, x is not.
//	if (y1 == y2)
//	{
//		Q11 = frame.at<cv::Vec3b>(y1, x1);
//		Q21 = frame.at<cv::Vec3b>(y1, x2);
//		rgb[0] = (Q11[0] * (x2 - x) + Q21[0] * (x - x1)) / (x2 - x1);
//		rgb[1] = (Q11[1] * (x2 - x) + Q21[1] * (x - x1)) / (x2 - x1);
//		rgb[2] = (Q11[2] * (x2 - x) + Q21[2] * (x - x1)) / (x2 - x1);
//		return true;
//	}
//
//	// When x and y are not intergers
//	Q11 = frame.at<cv::Vec3b>(y1, x1);
//	Q12 = frame.at<cv::Vec3b>(y2, x1);
//	Q21 = frame.at<cv::Vec3b>(y1, x2);
//	Q22 = frame.at<cv::Vec3b>(y2, x2);
//
//	// Compute the RGB at (x, y)
//	rgb[0] = (Q11[0] * (x2 - x)*(y2 - y) + Q21[0] * (x - x1)*(y2 - y) + Q12[0] * (x2 - x)*(y - y1) + Q22[0] * (x - x1)*(y - y1)) / ((x2 - x1)*(y2 - y1));
//	rgb[1] = (Q11[1] * (x2 - x)*(y2 - y) + Q21[1] * (x - x1)*(y2 - y) + Q12[1] * (x2 - x)*(y - y1) + Q22[1] * (x - x1)*(y - y1)) / ((x2 - x1)*(y2 - y1));
//	rgb[2] = (Q11[2] * (x2 - x)*(y2 - y) + Q21[2] * (x - x1)*(y2 - y) + Q12[2] * (x2 - x)*(y - y1) + Q22[2] * (x - x1)*(y - y1)) / ((x2 - x1)*(y2 - y1));
//
//	return true;
//}
//
//
//static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r)
//{
//
//	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right(), r.bottom()));
//
//}
//
//static void calculateDelaunayTriangles(cv::Rect rect, std::vector<cv::Point2f> &points, std::vector< std::vector<int> > &delaunayTri) {
//
//	// Create an instance of Subdiv2D
//	cv::Subdiv2D subdiv(rect);
//
//	// Insert points into subdiv
//	for (std::vector<cv::Point2f>::iterator it = points.begin(); it != points.end(); it++)
//	{
//		/*cv::Point2f p = *it;
//		if (0 <= p.x && p.x <= rect.width && 0 <= p.y && p.y < rect.height)*/
//			subdiv.insert(*it);
//	}
//	std::vector<cv::Vec6f> triangleList;
//	subdiv.getTriangleList(triangleList);
//	std::vector<cv::Point2f> pt(3);
//	std::vector<int> ind(3);
//
//	for (size_t i = 0; i < triangleList.size(); i++)
//	{
//		cv::Vec6f t = triangleList[i];
//		pt[0] = cv::Point2f(t[0], t[1]);
//		pt[1] = cv::Point2f(t[2], t[3]);
//		pt[2] = cv::Point2f(t[4], t[5]);
//
//		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])) {
//			for (int j = 0; j < 3; j++)
//				for (size_t k = 0; k < points.size(); k++)
//					if (abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
//						ind[j] = k;
//
//			delaunayTri.push_back(ind);
//		}
//	}
//
//}
//
//// Apply affine transform calculated using srcTri and dstTri to src
//void applyAffineTransform(cv::Mat &warpImage, cv::Mat &src, std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri)
//{
//	// Given a pair of triangles, find the affine transform.
//	cv::Mat warpMat = getAffineTransform(srcTri, dstTri);
//
//	// Apply the Affine Transform just found to the src image
//	warpAffine(src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
//}
//
//
//void warpTriangle(cv::Mat &img1, cv::Mat &img2, std::vector<cv::Point2f> &t1, std::vector<cv::Point2f> &t2, cv::Mat &blank)
//{
//
//	cv::Rect r1 = boundingRect(t1);
//	cv::Rect r2 = boundingRect(t2);
//
//	// Offset points by left top corner of the respective rectangles
//	std::vector<cv::Point2f> t1Rect, t2Rect;
//	std::vector<cv::Point> t2RectInt;
//	for (int i = 0; i < 3; i++)
//	{
//
//		t1Rect.push_back(cv::Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
//		t2Rect.push_back(cv::Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
//		t2RectInt.push_back(cv::Point(t2[i].x - r2.x, t2[i].y - r2.y)); // for fillConvexPoly
//
//	}
//
//	// Get mask by filling triangle
//	cv::Mat mask = cv::Mat::zeros(r2.height, r2.width, CV_32FC3);
//	fillConvexPoly(mask, t2RectInt, cv::Scalar(1.0, 1.0, 1.0), 16, 0);
//
//	// Apply warpImage to small rectangular patches
//	cv::Mat img1Rect;
//	img1(r1).copyTo(img1Rect);
//
//	cv::Mat img2Rect = cv::Mat::zeros(r2.height, r2.width, img1Rect.type());
//
//	applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);
//
//	multiply(img2Rect, mask, img2Rect);
//	multiply(img2(r2), cv::Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
//	multiply(blank(r2), cv::Scalar(1.0, 1.0, 1.0) - mask, blank(r2));
//	/*img2(r2) = img2(r2) + img2Rect;*/
//	blank(r2) = blank(r2) + img2Rect;
//
//
//
//}
//
//static int N = 0;
//
//int main(int argc, char * argv[])
//{
//	
//
//	//Loads happy and sad images
//	cv::Mat img_happy = cv::imread("D:\\Projects\\CV_project\\opcv_test\\face_landmark\\data\\happy.jpg", 1);
//	cv::Mat img_sad = cv::imread("D:\\Projects\\CV_project\\opcv_test\\face_landmark\\data\\sad.jpg", 1);
//	
//	//Reads the image size 
//	cv::Size imgSz = cv::imread("D:\\Projects\\CV_project\\opcv_test\\face_landmark\\data\\happy00.png", 0).size();
//	//Calculate the vector length for one image
//	N = imgSz.width * imgSz.height;
//	//cv::Mat projctedTrain;
//	//cv::Mat projectionMatrix;
//	//PCA(N, projctedTrain, projectionMatrix);
//	//Training images
//	cv::Mat S = cv::Mat::zeros(N, 1, CV_32FC1);
//	cv::Mat I = cv::Mat(N, M, CV_32FC1);
//	char file[64];
//	for (int i = 0; i < M / 2; i++)
//	{
//		sprintf(file, "D:\\Projects\\CV_project\\opcv_test\\face_landmark\\data\\happy%02d.png", i);
//		cv::Mat m = cv::imread(file, CV_LOAD_IMAGE_GRAYSCALE);
//		m = m.t();
//		m = m.reshape(1, N);
//		m.convertTo(m, CV_32FC1);
//		m.copyTo(I.col(i));
//		S = S + m;
//	}
//	for (int i = 0; i < M / 2; i++)
//	{
//		sprintf(file, "D:\\Projects\\CV_project\\opcv_test\\face_landmark\\data\\sad%02d.png", i);
//		cv::Mat m = cv::imread(file, CV_LOAD_IMAGE_GRAYSCALE);
//		m = m.t();
//		m = m.reshape(1, N);
//		m.convertTo(m, CV_32FC1);
//		m.copyTo(I.col(i + M / 2));
//		S = S + m;
//	}
//
//	//calculate eigenvectors
//	cv::Mat mean = S / (float)M;
//	cv::Mat A = cv::Mat(N, M, CV_32FC1);
//	for (int i = 0; i < M; i++)
//	{
//		A.col(i) = I.col(i) - mean;
//	}
//	cv::Mat C = A.t() * A;
//	cv::Mat EVal, EVec;
//	cv::eigen(C, EVal, EVec);
//
//	//compute projection matrix 
//	cv::Mat U = A * EVec;
//	cv::Mat projectionMatrix = U.t();
//
//	// project the training set to the faces space
//	cv::Mat projctedTrain = projectionMatrix * A;
//
//
//
//	//start camera
//	cv::VideoCapture vc;
//	char ch;
//	cv::Mat frame;
//	bool isVideo = true;
//
//	//setup dlib
//	frontal_face_detector detector = get_frontal_face_detector();
//	shape_predictor pose_model;
//
//
//	// Output image is set to white
//	//Mat imgOut = Mat::ones(imgIn.size(), imgIn.type());
//	/*cv::Mat imgOut = cv::imread("D:\\Projects\\CV_project\\opcv_test\\face_landmark\\data\\after.png");
//	cv::Size sizeOut = imgOut.size();
//	cv::Rect rectOut(0, 0, sizeOut.width, sizeOut.height);
//
//	std::vector<rectangle> face_before = detector(imgIn);
//	std::vector<rectangle> face_after = detector(imgOut);
//	std::vector<cv::Point2f> points_bf,points_af;
//	for (int i = 0; i < 68; i++)
//	{
//		points_bf.push_back(cv::Point2f(pose_model(imgIn, face_before[0]).part(i).x(), pose_model(imgIn, face_before[0]).part(i).y()));
//		points_af.push_back(cv::Point2f(pose_model(imgOut, face_after[0]).part(i).x(), pose_model(imgOut, face_after[0]).part(i).y()));
//	}
//	std::vector<int> hull_bf_index;
//	std::vector<cv::Point2f> hull_bf_point, hull_af_point;
//	cv::convexHull(points_bf, hull_bf_index, false, false);
//	for (int i = 0; i < hull_bf_index.size(); i++)
//	{
//		hull_bf_point.push_back(points_bf[hull_bf_index[i]]);
//		hull_af_point.push_back(points_af[hull_bf_index[i]]);
//	}
//	cv::Rect rect(0, 0, imgIn.cols, imgIn.rows);
//	std::vector< std::vector<int> > dt1;
//	calculateDelaunayTriangles(rect, hull_bf_point, dt1);*/
//
//
//	int happy_count = 0;
//	int sad_count = 0;
//
//	try {
//		vc.open(0);//open default camera
//		deserialize("D:\\Projects\\CV_project\\opcv_test\\face_landmark\\shape_predictor_68_face_landmarks.dat") >> pose_model;
//	}
//	catch (cv::Exception e) {
//		frame = cv::Mat::eye(100, 100, CV_32FC1);
//		isVideo = false;
//	}
//	catch (serialization_error& e)
//	{
//		cout << "you need dlib's default face landmarking model file to run this example." << endl;
//		cout << "you can get it from the following url: " << endl;
//		cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
//		cout << endl << e.what() << endl;
//	}
//	catch (exception& e)
//	{
//		cout << e.what() << endl;
//	}
//	
//	while ((ch = cv::waitKey(10)) != 'q')
//	{
//		if (isVideo) {
//			try
//			{
//				vc >> frame;
//				cv_image<bgr_pixel> cimg(frame);
//
//				// detect the keyboard
//				int c = cv::waitKey(50);
//				
//
//				// detect faces
//				std::vector<rectangle> faces = detector(cimg);
//				// get the biggest face
//				rectangle maxFace;
//				for (int i = 0; i < faces.size(); i++)
//				{
//					if (faces[i].area() > maxFace.area()) maxFace = faces[i];
//				}
//
//
//
//				// convert dlib rectangle to opencv rect
//				cv::Rect *roi = 0;
//				roi = &dlibRectangleToOpenCV(maxFace);
//
//				// get landmarks on mouth (#46~#67)
//				std::vector<cv::Point2f> mouthLM;
//				for (int i = 48; i < 68 ; i++)
//				{
//					mouthLM.push_back(cv::Point2f(pose_model(cimg, maxFace).part(i).x(),
//						pose_model(cimg, maxFace).part(i).y()));
//				}
//				
//				cv::Rect rectMouth = cv::boundingRect(mouthLM);
//				
//				//for testing
//				/*cv::rectangle(frame, rectMouth.tl(), rectMouth.br(), (255, 255, 255), 3);
//				cv::imshow("face", frame);*/
//				
//				int suby = roi->height * 0.5;
//				roi->height -= suby;
//				roi->y += suby;
//				int subx = (roi->width - roi->height) / 2 * 0.7;
//				roi->width -= subx * 2;
//				roi->x += subx;
//				cv::Mat imgROI = frame(*roi);
//				cv::Mat *subimg = &cv::Mat(100, 100 * 0.7, CV_8UC3);
//				cv::Mat *subimg_gray = &cv::Mat(100, 100 * 0.7, CV_8UC1);
//				cv::Size size(100, 100 * 0.7);
//				//cv::resize(frame, *subimg, size);
//				cv::resize(imgROI, *subimg, size);
//				cv::cvtColor(*subimg, *subimg_gray, cv::COLOR_BGR2GRAY);
//				cv::equalizeHist(*subimg_gray, *subimg_gray);
//				// getting training data
//				/*switch (c)
//				{
//					char file[32];
//					case 'h':
//						sprintf(file, "D:\\Projects\\CV_project\\opcv_test\\face_landmark\\data\\happy%02d.png", happy_count);
//						cv::imwrite(file, *subimg_gray);
//						happy_count++;
//						cv::waitKey(1000);
//						break;
//					case 's':
//						sprintf(file, "D:\\Projects\\CV_project\\opcv_test\\face_landmark\\data\\sad%02d.png", sad_count);
//						cv::imwrite(file, *subimg_gray);
//						sad_count++;
//						cv::waitKey(1000);
//						break;
//				}*/
//
//				// recognize mood
//				double min = 0.0; 
//				int min_ind;
//				cv::Mat subimgGray;
//				subimg_gray->copyTo(subimgGray);
//				subimgGray = subimgGray.t();
//				subimgGray = subimgGray.reshape(1, N);
//				subimgGray.convertTo(subimgGray, CV_32FC1);
//				cv::Mat proj = projectionMatrix * subimgGray;
//				// find the minimum distance vector
//				for (int i = 0; i < M; i++)
//				{
//					double n = norm(proj - projctedTrain.col(i));
//					if (min > n || i == 0)
//					{
//						min = n;
//						min_ind = i;
//					}
//				}
//				/*if (min_ind > M/2) cv::imshow("logo", img_sad);
//				else cv::imshow("logo", img_happy);
//				cv::moveWindow("logo", 670, 200);*/
//
//				/*cv::imshow("face", frame);
//				cv::moveWindow("face", 0, 0);
//
//				cv::imshow("gray", *subimg_gray);
//				cv::moveWindow("gray", 670, 0);*/
//
//				// happy = 0, sad = 1
//				int mood = 0;
//				if (min_ind > M / 2) mood = 1;
//					
//
//				cv::Mat curMouth, curFrame, output;
//				frame.copyTo(output);
//				if (mood == 0)
//				{
//					int w_frame = frame.cols;
//					int h_frame = frame.rows;
//					int w = rectMouth.width; // Change this to mouth max width
//					int h = rectMouth.height; // Change this to mouth max height
//					int x0 = rectMouth.x; // Change this to left most x of mouth contour 
//					int y0 = rectMouth.y; // Change this to up most y of mouth contour 
//										 // Giving the bounding box
//					int r0 = y0 - h / 2;
//					int rt = y0 + 3 * h / 2;
//					int c0 = x0;
//					int ct = x0 + w;
//					int cm = (c0 + ct) / 2;
//					int sc = (rt - r0) / 4;
//
//					cv::Mat curFace(2 * h, w, CV_8UC3);
//					for (int r = 0; r < rt - r0; r++) {
//						for (int c = 0; c < ct - c0; c++) {
//							curFace.at<cv::Vec3b>(r, c)[0] = 0;
//							curFace.at<cv::Vec3b>(r, c)[1] = 255;
//							curFace.at<cv::Vec3b>(r, c)[2] = 0;
//
//							double x2 = 0.0;
//							double y2 = 0.0;
//
//							//Project((double)c, (double)r, x2, y2, homInv);
//							x2 = c + c0;
//							y2 = r + r0 + sc*cos(double((x2 - cm)) / (ct - cm)*M_PI);
//							double rgb2[3];
//							if (x2 < w_frame && y2 < h_frame && x2 >= 0 && y2 >= 0) {
//								BilinearInterpolation(frame, x2, y2, rgb2);
//								if (rgb2[0] != 0 || rgb2[1] != 0 || rgb2[2] != 0) {
//									curFace.at<cv::Vec3b>(r, c)[0] = rgb2[0];
//									curFace.at<cv::Vec3b>(r, c)[1] = rgb2[1];
//									curFace.at<cv::Vec3b>(r, c)[2] = rgb2[2];
//								}
//							}
//
//						}
//					}
//					
//					curFace.copyTo(curMouth);
//					curMouth.convertTo(curMouth, CV_32F);
//					
//					frame.copyTo(curFrame);
//					curFrame.convertTo(curFrame, CV_32F);
//				
//					multiply(curFrame(rectMouth), cv::Scalar(0.0, 0.0, 0.0), curFrame(rectMouth));
//
//					cv::Mat curMouth_sub = cv::Mat(rectMouth.size(), CV_32F);
//					cv::resize(curMouth, curMouth_sub, rectMouth.size());
//					curFrame(rectMouth) = curFrame(rectMouth) + curMouth_sub;
//					curFrame.convertTo(curFrame, CV_8UC3);
//					
//					curFrame.copyTo(output);
//				
//					
//				}
//				
//
//				cv::imshow("deform", output);
//				cv::moveWindow("deform", 0, 0);
//				if (mood == 1) cv::imshow("logo", img_sad);
//				else cv::imshow("logo", img_happy);
//				cv::moveWindow("logo", 670, 200);
//
//			}
//			catch (const std::exception& e)
//			{
//				cout << e.what() << endl;
//			}
//		}
//	}
//	if (isVideo)
//		vc.release();
//	return 0;
//}
//
