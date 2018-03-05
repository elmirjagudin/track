#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <unistd.h>

using namespace cv;
using namespace std;

void
find_matches(Mat &old_desc, Mat &curr_desc, vector<DMatch> &matches)
{
    vector<DMatch> all_matches;
    auto bf = BFMatcher(NORM_HAMMING, true);
    bf.match(old_desc, curr_desc, all_matches);
    printf("%ld matches\n", matches.size());

    matches = vector<DMatch>();

    for (auto i = all_matches.begin(); i != all_matches.end(); ++ i)
    {
        auto ma = *i;
        if (ma.distance <= 10)
        {
            matches.push_back(ma);
        }
    }
}

void save_frame(Mat &old_desc, Mat &curr_desc,
                Mat &old_frame, Mat &curr_frame,
                vector<KeyPoint> &old_keypoints, vector<KeyPoint> &keypoints)
{
    old_desc = curr_desc.clone();
    old_frame = curr_frame.clone();
    old_keypoints = keypoints;
}

void
skip_frames(VideoCapture &cap, int to_skip)
{
   Mat dummy;
   for (int i = to_skip; i > 0; i -= 1)
   {
       cap >> dummy;
       printf("i %d\n", i);
   }
}

int
main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: %s <Image_Path>\n", argv[0]);
        return -1;
    }
    auto check_matches = false;
    auto orb = cv::ORB::create(128*2);
    vector<KeyPoint> old_keypoints;
    vector<KeyPoint> keypoints;
    Mat tmp;
    Mat frame;
    Mat old_frame;
    Mat desciptors;
    Mat old_desciptors;
    vector<DMatch> matches;

    VideoCapture cap(argv[1]);

    if (!cap.isOpened())
    {
      printf("Error opening video stream or file\n");
      return -1;
    }

    skip_frames(cap, 50);

    while (1)
    {
       /* get next frame */
       cap >> tmp;
  
       /* no more frames */
       if (tmp.empty())
       {
           break;
       }

       cv::cvtColor(tmp, frame, COLOR_BGR2GRAY);

       orb->detectAndCompute(_InputArray(frame), cv::noArray(), keypoints, desciptors);

//       std::vector<int>::iterator itr;
//       printf("-----------------------------------------------------------");
//       for (auto itr = keypoints.begin(); itr != keypoints.end(); ++itr )
//       {
//           auto kp = *itr;
//           printf("kp %f %f\n", kp.pt.x, kp.pt.y);
//       }

        if (check_matches)
        {
            find_matches(old_desciptors, desciptors, matches);
            drawMatches(old_frame, old_keypoints, frame, keypoints, matches, tmp);

               Mat small;
               resize(tmp, small, Size(), 0.4, 0.5, INTER_LANCZOS4);
               imshow("Frame", small);
               waitKey();



//            for (auto i = matches.begin(); i != matches.end(); ++ i)
//            {
//                auto ma = *i;
//                if (ma.distance > 10) {
//                  /* skip it */
//                  continue;
//                }
//                printf("distance %f\n", ma.distance);

//               vector<DMatch> m = vector<DMatch>(1);
//               m[0] = ma;
//               drawMatches(old_frame, old_keypoints, frame, keypoints, m, tmp);
//               Mat small;
//               resize(tmp, small, Size(), 0.5, 0.5, INTER_LANCZOS4);
//               imshow("Frame", small);
//               waitKey(50);
//            }

        }     


//        cv::Mat img_kpts;
//        cv::drawKeypoints(frame, keypoints, img_kpts, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
//        imshow("Frame", img_kpts );


        save_frame(old_desciptors, desciptors, old_frame, frame, old_keypoints, keypoints);
        check_matches = true;

       
//printf("keypoints %ld\n", keypoints.size());

 
       // Display the resulting frame
 //      imshow("Frame", frame );
 
     // Press  ESC on keyboard to exit
//     if (waitKey() == 27)
//     {
//       break;
//     }
  }
    return 0;

}
