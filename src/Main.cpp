/*
----------------------------------------------
--- Author         : Ahmet Özlü
--- Mail           : ahmetozlu93@gmail.com
--- Date           : 1st August 2017
--- Version        : 1.0
--- OpenCV Version : 2.4.10
--- Demo Video     : https://youtu.be/3uMKK28bMuY
----------------------------------------------
*/
using namespace std;
#include "Blob.h"
#include <fstream>
#include <string>
#include <iomanip>
#pragma warning(disable : 4996)
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#define SHOW_STEPS // un-comment | comment this line to show steps or not

// const global variables
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

// function prototypes
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs);
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex);
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);
double distanceBetweenPoints(cv::Point point1, cv::Point point2);
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy);
void drawCarCountOnImage(int &carCountRight, cv::Mat &imgFrame2Copy);

// global variables
std::stringstream date;
int carCountLeft, intVerticalLinePosition, carCountRight = 0;

int main(void) {				    
    cv::VideoCapture capVideo;
    cv::Mat imgFrame1;
    cv::Mat imgFrame2;
    std::vector<Blob> blobs;	

    //capVideo.open("/Users/collins/OSI/video_processing/mouse.avi");
    //capVideo.open("/Users/collins/OSI/video_processing/rats.avi");
    //capVideo.open("/Users/collins/OSI/video_processing/short_mouse.avi");
    //capVideo.open("/Users/collins/OSI/video_processing/2mice.avi");
    capVideo.open("../short_mouse.avi");

    if (!capVideo.isOpened()) {                                                 // if unable to open video file
        std::cout << "error reading video file" << std::endl << std::endl;      // show error message
        return(0);                                                              // and exit program
    }

    if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
        std::cout << "error: video file must have at least two frames";
        return(0);
    }

    /* MLC adds video output */
        std::cout << "Frame size is: " << unsigned(capVideo.get(3)) << "x" << unsigned(capVideo.get(4)) << std::endl << std::endl;      


    cv::VideoWriter outputVideo;                    // Open the output
    int ex = static_cast<int>(capVideo.get(CV_CAP_PROP_FOURCC));
    cv::Size S = cv::Size((int) capVideo.get(CV_CAP_PROP_FRAME_WIDTH), (int) capVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
    const string NAME = "output.avi";
    const bool askOutputType = 0; // If false it will use the inputs codec type
    if (askOutputType)
        outputVideo.open(NAME, ex=-1, capVideo.get(CV_CAP_PROP_FPS), S, true);
    else
        outputVideo.open(NAME, CV_FOURCC('M','J','P','G'), capVideo.get(CV_CAP_PROP_FPS), S, true);

    if (!outputVideo.isOpened())
    {
        std::cout << "Could not open output video for write:" << std::endl << std::endl;      // show error message
        return -1;
    }







    bool ret = true;
   ret =  capVideo.read(imgFrame1);
    if (ret == false) { return(0);}
    ret = capVideo.read(imgFrame2);
    if (ret == false) { return(0);}




    char chCheckForEscKey = 0;
    bool blnFirstFrame = true;
    int frameCount = 2;
    


    /*Ahmet's algorithm explanation
     *
     * Convert the source image to binary images by applying thresholding with 
     * several thresholds from minThreshold (inclusive) to maxThreshold (exclusive)
     * with distance thresholdStep between neighboring thresholds.
     * 
     * Extract connected components from every binary image by findContours and calculate their centers.
     * 
     * Group centers from several binary images by their coordinates. Close centers form one group that
     * corresponds to one blob, which is controlled by the minDistBetweenBlobs parameter.
     * 
     * From the groups, estimate final centers of blobs and their radiuses and return as locations and sizes of keypoints.*/

    while (capVideo.isOpened() && chCheckForEscKey != 27) {
        std::vector<Blob> currentFrameBlobs;
        cv::Mat imgFrame1Copy = imgFrame1.clone();
        cv::Mat imgFrame2Copy = imgFrame2.clone();
        cv::Mat imgDifference;
        cv::Mat imgThresh;
        cv::cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
        cv::cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);
        cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
        cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);
        cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);
        // Base threshold is difference between frames -MC
        cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);
        cv::imshow("imgThresh", imgThresh);
        cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

        // Loop dilates and erodes the threshold several times. A smoothing process
        // with a tendeancy to dilate features and connect disjointed sections.
        // 5 iterations is good for Pishan's video - MC
        for (unsigned int i = 0; i < 5; i++) {
            cv::dilate(imgThresh, imgThresh, structuringElement5x5);
            cv::dilate(imgThresh, imgThresh, structuringElement5x5);
            cv::erode(imgThresh, imgThresh, structuringElement5x5);
        }

        /* If above thresholding operations are insufficient, we can add a second operation
         * here. It could act on a still frame to discriminate between light and dark patches.
         * Rodents are either white or black. The difference threshold could be cross referenced 
         * against the solid-patch/light-dark threshold to improve results. -MC*/
        cv::Mat imgThreshCopy = imgThresh.clone();
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
		drawAndShowContours(imgThresh.size(), contours, "imgContours");

        std::vector<std::vector<cv::Point> > convexHulls(contours.size());

        // Convex Hulls draw an envelope around the contours. That is, since mice are solid,
        // only the edges will be detected frame-to-frame. Convex hulls fills in their middle. -MC
        for (unsigned int i = 0; i < contours.size(); i++) {
            cv::convexHull(contours[i], convexHulls[i]);
        }

        drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");

        /* Each convex hull is evaluated for its worth as a blob. Parameters like 
         * area and aspect ratio establish the convex hull's blobiness. Those worthy
         * are included in the list of currentFrameBlobs -MC */
        for (auto &convexHull : convexHulls) {
            Blob possibleBlob(convexHull);

            // Try adding a convexity metric - MC
            // See Blob.h for more ideas - MC
            if (possibleBlob.currentBoundingRect.area() > 600 &&
                possibleBlob.dblCurrentAspectRatio > 0.3 &&
                possibleBlob.dblCurrentAspectRatio < 3.0 &&
                possibleBlob.currentBoundingRect.width > 50 &&
                possibleBlob.currentBoundingRect.height > 50 &&
                possibleBlob.dblCurrentDiagonalSize > 70.0 &&
                (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50) {
					currentFrameBlobs.push_back(possibleBlob);
            }
        }

        drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");

        if (blnFirstFrame == true) {
            for (auto &currentFrameBlob : currentFrameBlobs) {
                blobs.push_back(currentFrameBlob);
            }
        } 
		else {
            // Handles blob tracking frame-to-frame. Moves currentFrameBlobs to blobs.
            // Includes parameters such as match distance and tracking time. - MC
            matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
        }

        drawAndShowContours(imgThresh.size(), blobs, "imgBlobs");


        /* Does counting */
        // Counting section removed

        imgFrame2Copy = imgFrame2.clone();	// get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

        drawBlobInfoOnImage(blobs, imgFrame2Copy);


        cv::imshow("imgFrame2Copy", imgFrame2Copy);
        outputVideo.write(imgFrame2Copy);
        /* END counting section */

        //cv::waitKey(0);	// uncomment this line to go frame by frame for debugging        
		
        // now we prepare for the next iteration
        currentFrameBlobs.clear();

        imgFrame1 = imgFrame2.clone();	// move frame 1 up to where frame 2 is

        if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {
            ret = capVideo.read(imgFrame2);
            if (ret == false) {return(0);}
        }
        else {
            std::cout << "end of video\n";
            break;
        }

        blnFirstFrame = false;
        frameCount++;
        chCheckForEscKey = cv::waitKey(1);
    }

    if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
        cv::waitKey(0);                         // hold the windows open to allow the "end of video" message to show
    }

    // note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows
    return(0);
}


void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs) {
    for (auto &existingBlob : existingBlobs) {
        existingBlob.blnCurrentMatchFoundOrNewBlob = false;
        existingBlob.predictNextPosition();
    }

    for (auto &currentFrameBlob : currentFrameBlobs) {
        int intIndexOfLeastDistance = 0;
        double dblLeastDistance = 100000.0;

        for (unsigned int i = 0; i < existingBlobs.size(); i++) {

            if (existingBlobs[i].blnStillBeingTracked == true) {
                double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

                if (dblDistance < dblLeastDistance) {
                    dblLeastDistance = dblDistance;
                    intIndexOfLeastDistance = i;
                }
            }
        }

        if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 1) {
            addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
        }
        else {
            addNewBlob(currentFrameBlob, existingBlobs);
        }

    }

    for (auto &existingBlob : existingBlobs) {
        if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) {
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
        }
        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 65) {
            existingBlob.blnStillBeingTracked = false;
        }
    }
}


void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {
    existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
    existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;
    existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());
    existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
    existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;
    existingBlobs[intIndex].blnStillBeingTracked = true;
    existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}


void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {
    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;
    existingBlobs.push_back(currentFrameBlob);
}


double distanceBetweenPoints(cv::Point point1, cv::Point point2) {    
    int intX = abs(point1.x - point2.x);
    int intY = abs(point1.y - point2.y);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}


void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {    
	cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);
    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);
    cv::imshow(strImageName, image);
}


void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) {    
    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);
    std::vector<std::vector<cv::Point> > contours;

    for (auto &blob : blobs) {
        if (blob.blnStillBeingTracked == true) {
            contours.push_back(blob.currentContour);
        }
    }

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);
    cv::imshow(strImageName, image);
}





void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy) {
    for (unsigned int i = 0; i < blobs.size(); i++) {
        if (blobs[i].blnStillBeingTracked == true) {
            cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);
            
			int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
			double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
            int intFontThickness = (int)std::round(dblFontScale * 1.0);

            cv::putText(imgFrame2Copy, std::to_string(i), blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
        }
    }
}


void drawCarCountOnImage(int &carCountRight, cv::Mat &imgFrame2Copy) {
    int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
    double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 450000.0;
    int intFontThickness = (int)std::round(dblFontScale * 2.5);
	
	// Right way
	cv::Size textSize = cv::getTextSize(std::to_string(carCountRight), intFontFace, dblFontScale, intFontThickness, 0);
	cv::putText(imgFrame2Copy, "Vehicle count:" + std::to_string(carCountRight), cv::Point(568,25), intFontFace, dblFontScale, SCALAR_RED, intFontThickness);

	// Left way
	cv::Size textSize1 = cv::getTextSize(std::to_string(carCountLeft), intFontFace, dblFontScale, intFontThickness, 0);
	cv::putText(imgFrame2Copy, "Vehicle count:" + std::to_string(carCountLeft), cv::Point(10, 25), intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);
}


