#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: %s <Image_Path>\n", argv[0]);
        return -1;
    }

    VideoCapture cap(argv[1]);

    if (!cap.isOpened())
    {
      printf("Error opening video stream or file\n");
      return -1;
    }

    while(1)
    {
       Mat frame;
       // Capture frame-by-frame
       cap >> frame;
  
       // If the frame is empty, break immediately
       if (frame.empty())
       {
           break;
       }
 
       // Display the resulting frame
       imshow("Frame", frame );
 
     // Press  ESC on keyboard to exit
     if(waitKey(1) == 27)
    {
      break;
    }
  }
    return 0;

}
