#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

#define FEATURES 16

class KnownPoints
{
    Mat descriptors;
    vector<KeyPoint> keypoints;
  public:
    KnownPoints()
    {
    }

    Mat & get_descriptors()
    {
        return this->descriptors;
    }

    void add_points(vector<KeyPoint> & keypoints,
                    Mat & descriptors,
                    vector<bool> & matched_idx,
                    vector<KeyPoint> & matched_keypoints)
    {
        matched_keypoints.clear();

        for (auto i = 0; i < matched_idx.size(); i += 1)
        {
            if (!matched_idx[i])
            {
                this->descriptors.push_back(descriptors.row(i));
                this->keypoints.push_back(keypoints[i]);
                matched_keypoints.push_back(keypoints[i]);
            }
        }
    }
};

bool
get_frame(VideoCapture &video, Mat &frame)
{
    Mat tmp;

    video >> tmp;
    if (tmp.empty())
    {
        return false;
    }

    cv::cvtColor(tmp, frame, COLOR_BGR2GRAY);
    return true;
}

void
get_keypoints(Mat & frame,
              vector<KeyPoint> & keypoints,
              Mat & descriptors)
{
    static auto orb = cv::ORB::create(FEATURES);
    orb->detectAndCompute(_InputArray(frame), cv::noArray(), keypoints, descriptors);
}

void
filter_matches(int num_keypoints, vector<DMatch> &matches, vector<bool> &matched_idx)
{
    matched_idx = vector<bool>(num_keypoints);

    for (auto i = matches.begin(); i != matches.end(); ++ i)
    {
        auto ma = *i;

        if (ma.distance <= 10)
        {
            matched_idx[ma.trainIdx] = true;
        }
    }
}

void
match_keypoints(vector<KeyPoint> & keypoints,
                Mat & descriptors,
                vector<KeyPoint> & matched_keypoints)
{
    static auto matcher = BFMatcher(NORM_HAMMING, true);
    static auto known_points = KnownPoints();

    auto matches = vector<DMatch>();

    matcher.match(known_points.get_descriptors(),
                  descriptors,
                  matches);

    vector<bool> matched_idx;
    filter_matches(keypoints.size(), matches, matched_idx);
    known_points.add_points(keypoints, descriptors, matched_idx, matched_keypoints);

//    cout << "matched " << matches.size() << "/" << keypoints.size() << endl;
}

void
show_matches(Mat & frame, vector<KeyPoint> matched_keypoints)
{
    cv::Mat img_kpts;
    cv::drawKeypoints(frame, matched_keypoints, img_kpts, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    Mat small;
    resize(img_kpts, small, Size(), 0.6, 0.6, INTER_LANCZOS4);

    imshow("Frame", small);
    waitKey(25);
}

void
process(VideoCapture &video)
{
    Mat frame;
    vector<KeyPoint> keypoints;
    vector<KeyPoint> matched_keypoints;
    Mat descriptors;

    while (get_frame(video, frame))
    {
        get_keypoints(frame, keypoints, descriptors);
        match_keypoints(keypoints, descriptors, matched_keypoints);

        show_matches(frame, matched_keypoints);
    }
    cout << "done\n";
}

int
main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("usage: %s <video_path>\n", argv[0]);
        return -1;
    }

    VideoCapture cap(argv[1]);
    if (!cap.isOpened())
    {
        printf("Error opening video stream or file\n");
        return -1;
    }
    process(cap);

    return 0;
}