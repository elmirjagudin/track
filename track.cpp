#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

#define FEATURES 128*4
#define N_BEST_POINTS 16
#define MAX_POINTS 1024 * 3

class MatchPair
{
public:
    KeyPoint known_point;
    KeyPoint frame_point;
    string name;
};

class KnownPoints
{
    Mat descriptors;
    vector<KeyPoint> keypoints;
    vector<string> keypoints_names;

  private:
    void sort_points_on_response(vector<KeyPoint> & keypoints,
                                 vector<bool> & matched_idx,
                                 vector<size_t> & idx)
    {
        for (auto i = 0; i < matched_idx.size(); i += 1)
        {
            if (matched_idx[i])
            {
                /* continue */
                continue;
            }
            idx.push_back(i);
        }

        auto glambda = [keypoints](int l, int r)
        {
            return keypoints[l].response > keypoints[r].response;
        };
        sort(idx.begin(), idx.end(), glambda);
    }
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
                    vector<bool> & matched_idx)
    {
        if (this->keypoints.size() > MAX_POINTS)
        {
            cout << "no more data" << endl;
            return;
        }

        vector<size_t> sorted_keypoint_idx;
        sort_points_on_response(keypoints, matched_idx, sorted_keypoint_idx);

        cout << "adding new points\n";
        for (size_t i = 0; i < sorted_keypoint_idx.size() && i < N_BEST_POINTS; i++)
        {
            auto idx = sorted_keypoint_idx[i];
            this->descriptors.push_back(descriptors.row(idx));
            this->keypoints.push_back(keypoints[idx]);
            this->keypoints_names.push_back(to_string(this->keypoints.size()));
       }
    }

    void
    get_matched_point_pairs(vector<DMatch> & matches,
                            vector<KeyPoint> & keypoints,
                            vector<MatchPair> & pairs)
    {
        auto mp = MatchPair();
        mp.known_point = this->keypoints[0];
        mp.frame_point = keypoints[0];

        for (auto i = matches.begin(); i != matches.end(); i++)
        {
            auto ma = *i;

            auto mp = MatchPair();
            mp.name = this->keypoints_names[ma.queryIdx];
            mp.known_point = this->keypoints[ma.queryIdx];
            mp.frame_point = keypoints[ma.trainIdx];

            pairs.push_back(mp);
        }
    }

    string
    get_point_name(size_t idx)
    {
        return this->keypoints_names[idx];
    }
};

bool
get_frame(VideoCapture &video, Mat &frame)
{
    cout << " --------------------------------------------- " << endl;
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
filter_matches(int num_keypoints, vector<DMatch> &matches,
               vector<DMatch> &filtered_matches,
               vector<bool> &matched_idx)
{
    matched_idx = vector<bool>(num_keypoints);

    for (auto i = matches.begin(); i != matches.end(); ++ i)
    {
        auto ma = *i;
        matched_idx[ma.trainIdx] = true;

        if (ma.distance <= 16)
        {
            filtered_matches.push_back(ma);
        }
    }
}

void
match_keypoints(vector<KeyPoint> & keypoints,
                Mat & descriptors,
                vector<MatchPair> & pairs)
{
    static auto matcher = BFMatcher(NORM_HAMMING, true);
    static auto known_points = KnownPoints();
    static auto bootstrap = true;

    auto matches = vector<DMatch>();
    matcher.match(known_points.get_descriptors(),
                  descriptors,
                  matches);
//    cout << "matched " << matches.size() << "/" << keypoints.size() << endl;

    vector<bool> matched_idx;

    auto filtered_matches = vector<DMatch>();
    filter_matches(keypoints.size(), matches, filtered_matches, matched_idx);

    cout << "filtered_matches " << filtered_matches.size() << endl;
    if (filtered_matches.size() > 0 || bootstrap)
    {
        known_points.add_points(keypoints, descriptors, matched_idx);
        known_points.get_matched_point_pairs(filtered_matches,
                                             keypoints,
                                             pairs);
        bootstrap = false;
    }
    else
    {
        cout << "lost track\n";
    }
}

void
next_step_delay()
{
    static int delay = -1;
    switch (waitKey(delay))
    {
        case 32:
            delay *= -1;
            break;
    }
}

void
show_matches(Mat & frame,
             vector<MatchPair> & pairs)
{
    cv::Mat img_kpts;
//    cv::drawKeypoints(frame, matched_keypoints, img_kpts, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    Point textPos = Point(10, 24);
    cout << "got " << pairs.size() << " pairs\n";
    for (auto i = pairs.begin(); i != pairs.end(); i++)
    {
        auto pair = *i;
//        cout << "pair " << pair.name << endl;
        putText(frame, pair.name,
                textPos, FONT_HERSHEY_DUPLEX, 1,
                Scalar::all(-1));

        line(frame, textPos, pair.frame_point.pt, Scalar::all(-1));

        textPos.y += 24;
    }

    Mat small;
//    resize(img_kpts, small, Size(), 0.6, 0.6, INTER_LANCZOS4);

    imshow("Frame", frame);
    next_step_delay();
}

void
process(VideoCapture &video)
{
    Mat frame;
    vector<KeyPoint> keypoints;
    vector<KeyPoint> matched_keypoints;
    vector<string> matched_keypoints_names;
    Mat descriptors;
    vector<MatchPair> pairs;

    while (get_frame(video, frame))
    {
        get_keypoints(frame, keypoints, descriptors);
        pairs.clear();
        match_keypoints(keypoints, descriptors, pairs);

        show_matches(frame, pairs);
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