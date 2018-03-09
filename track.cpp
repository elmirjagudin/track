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

    void add_points(float vertical_offset,
                    vector<KeyPoint> & keypoints,
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
            keypoints[idx].pt.x += vertical_offset;
            this->keypoints.push_back(keypoints[idx]);
            this->keypoints_names.push_back(
                to_string(this->keypoints.size()) + " X " + to_string(keypoints[idx].pt.x));
       }
    }

    void
    get_matched_point_pairs(vector<DMatch> & matches,
                            vector<KeyPoint> & keypoints,
                            vector<MatchPair> & pairs,
                            float & vertical_offset)
    {
        double total_x_diff = 0;
        for (auto i = matches.begin(); i != matches.end(); i++)
        {
            auto ma = *i;

            auto mp = MatchPair();
            mp.name = this->keypoints_names[ma.queryIdx];
            mp.known_point = this->keypoints[ma.queryIdx];
            mp.frame_point = keypoints[ma.trainIdx];

            pairs.push_back(mp);
            total_x_diff += (mp.known_point.pt.x - mp.frame_point.pt.x);
        }

        if (matches.size() > 0)
        {
            vertical_offset = total_x_diff / matches.size();
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

    video >> frame;
    if (frame.empty())
    {
        return false;
    }

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
    vector<bool> matched_idx;

    auto filtered_matches = vector<DMatch>();
    filter_matches(keypoints.size(), matches, filtered_matches, matched_idx);

    if (filtered_matches.size() > 0 || bootstrap)
    {
        float vertical_offset = 0;
        known_points.get_matched_point_pairs(filtered_matches,
                                             keypoints,
                                             pairs,
                                             vertical_offset);

        cout << "vertical_offset " << vertical_offset << endl;
        known_points.add_points(vertical_offset, keypoints, descriptors, matched_idx);
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
save_image(Mat & frame)
{
    static int n = 0;

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    imwrite("frames/img" + to_string(++n) + ".png",
            frame,
            compression_params);
}

void
show_matches(Mat & frame,
             vector<MatchPair> & pairs)
{
    cv::Mat img_kpts;

    Point textPos = Point(10, 24);

    vector<Scalar> colors;
    colors.push_back(Scalar(0, 0, 255));
    colors.push_back(Scalar(0, 255, 0));
    colors.push_back(Scalar(255, 0, 0));
    size_t currColor = 0;

    for (auto i = pairs.begin(); i != pairs.end(); i++)
    {
        auto pair = *i;
        putText(frame, pair.name,
                textPos, FONT_HERSHEY_DUPLEX, 1,
                colors[currColor]);

        line(frame, textPos, pair.frame_point.pt, colors[currColor]);

        textPos.y += 24;

        currColor = (currColor + 1) % colors.size();
    }

//    Mat small;
//    resize(img_kpts, small, Size(), 0.6, 0.6, INTER_LANCZOS4);

    imshow("Frame", frame);
//    save_image(frame);
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