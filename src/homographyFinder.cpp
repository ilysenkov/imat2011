#include "homographyFinder.hpp"
#include <opencv2/opencv.hpp>
#include <functional>
#include <algorithm>

using namespace cv;
using namespace std;

struct WrongDmatch
{
  const static int wrong = -20;
  bool operator()(const DMatch& dm) const
  {
    return dm.imgIdx == wrong;
  }
};

void geometricConsistency(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
                          vector<DMatch>& matches1to2)
{
  const Size sz(300, 300);

  vector < Point2f > pts1;
  vector < Point2f > pts2;
  KeyPoint::convert(keypoints1, pts1);
  KeyPoint::convert(keypoints2, pts2);

  std::transform(pts2.begin(), pts2.end(), pts2.begin(), bind2nd(plus<Point2f> (), Point2f((float)sz.width, 0)));

  vector<float> dirs(matches1to2.size());
  for (size_t i = 0; i < matches1to2.size(); ++i)
  {
    DMatch& dm = matches1to2[i];
    Point2f line = pts2[dm.trainIdx] - pts1[dm.queryIdx];
    //line *= (float)1.0/norm(line);
    dirs[i] = cv::fastAtan2(line.y, line.x);
    if (dirs[i] > 180.0f)
      dirs[i] -= 360.0f;
    //dirs[i] = atan2(line.y, line.x);
  }

  Scalar mean, sdv;
  cv::meanStdDev(dirs, mean, sdv);

  const double minStdDev = 5;
  if (sdv.val[0] < minStdDev)
    return;

  for (size_t i = 0; i < matches1to2.size(); ++i)
  {
    if (fabs(dirs[i] - mean.val[0]) < sdv.val[0])
      continue;

    matches1to2[i].imgIdx = WrongDmatch::wrong;
  }
  matches1to2.erase(std::remove_if(matches1to2.begin(), matches1to2.end(), WrongDmatch()), matches1to2.end());
}

void orientationConsistencyFilter(const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
                                  vector<DMatch>& matches1to2, int angleDiff)
{
  // compute orientation differences histogramm
  vector<int> hist(180, 0);
  vector<int> binIndices(matches1to2.size());
  for (size_t mi = 0; mi < matches1to2.size(); mi++)
  {
    float angle1 = keypoints1[matches1to2[mi].queryIdx].angle;
    float angle2 = keypoints2[matches1to2[mi].trainIdx].angle;

    float maxa = max(angle1, angle2);
    float mina = min(angle1, angle2);

    float dist1 = maxa - mina;
    float dist2 = mina + 360 - maxa;

    float dist = min(dist1, dist2);

    int binIndex = (int)std::floor(dist);

    binIndices[mi] = binIndex;
    hist[binIndex]++;
  }

  int maxBinIndex = std::max_element(hist.begin(), hist.end()) - hist.begin();

  vector < DMatch > filteredMatches;
  for (size_t mi = 0; mi < matches1to2.size(); mi++)
  {
    if (std::abs(binIndices[mi] - maxBinIndex) <= angleDiff)
      filteredMatches.push_back(matches1to2[mi]);
  }
  std::swap(filteredMatches, matches1to2);
}

HomographyFinder::HomographyFinder()
{
  minSingularValue = 1e-4;
  inlierDistance = 4;
  lmInlierDistance = 6;
  ransacReprojectionError = 3;
  findHomographyIterationsCount = 10000;
  angleDiff = 5.0;

  maxInliersCount = 100;

  minHomographyPoints = 4;
}

bool HomographyFinder::isHomographyDegenerate(const Mat &H)
{
  Mat w, u, vt;
  SVD::compute(H, w, u, vt);
  double minv;
  cv::minMaxLoc(w, &minv);
  double eps = 1e-6;

#if !defined _MSC_VER
  return (isnan(minv) || minv < minSingularValue || fabs(determinant(vt.t() * u) + 1) < eps);
#else
  return (_isnan(minv) || minv < minSingularValue || fabs(determinant(vt.t() * u) + 1) < eps);
#endif

}

int HomographyFinder::getInliersCount(const cv::Mat &H, const std::vector<Point2f> &points1,
                                      const std::vector<Point2f> &points2, double distance, vector<int> *inliers)
{
  if (inliers != 0)
    inliers->clear();
  Mat points1t;
  perspectiveTransform(Mat(points1), points1t, H);

  int inliersCount = 0;
  for (size_t i1 = 0; i1 < points1.size(); i1++)
  {
    if (norm(points2[i1] - points1t.at<Point2f> ((int)i1, 0)) < distance) // inlier
    {
      inliersCount++;
      if (inliers != 0)
      {
        inliers->push_back(i1);
      }
    }
  }
  return inliersCount;
}

bool getRandomIndices(size_t count, int min, int max, vector<int> &indices)
{
  if (max - min < static_cast<int> (count))
    return false;

  indices.clear();
  while (indices.size() != count)
  {
    int cur = min + rand() % (max - min);
    if (std::find(indices.begin(), indices.end(), cur) == indices.end())
      indices.push_back(cur);
  }
  return true;
}

void filterPoints(const std::vector<cv::Point2f> &points, const std::vector<int> &inliers,
                  std::vector<cv::Point2f> &filteredPoints)
{
  filteredPoints.clear();
  filteredPoints.reserve(inliers.size());
  for (size_t i = 0; i < inliers.size(); i++)
  {
    filteredPoints.push_back(points[inliers[i]]);
  }
}

void getPoints(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
               const std::vector<cv::DMatch>& matches1to2, std::vector<Point2f> &points1, std::vector<Point2f> &points2)
{
  points1.clear();
  points2.clear();

  if (!matches1to2.empty())
  {
    vector<int> queryIdxs(matches1to2.size()), trainIdxs(matches1to2.size());
    for (size_t i = 0; i < matches1to2.size(); i++)
    {
      queryIdxs[i] = matches1to2[i].queryIdx;
      trainIdxs[i] = matches1to2[i].trainIdx;
    }

    KeyPoint::convert(keypoints1, points1, queryIdxs);
    KeyPoint::convert(keypoints2, points2, trainIdxs);
  }
}

void filterDumbbells(const std::vector<cv::Point2f> &points, std::vector<uchar> &isDumbbellMask)
{
  CV_Assert(isDumbbellMask.size() == points.size());

  double rSmall = 11;
  double rBig = 33;
  const int minPointsInBig = 2;
  const int minPointsInSmall = minPointsInBig;

  const float maxDumbellRatio = 1.25f;

  BruteForceMatcher < L2<float> > bruteForceMatcher;

  for (size_t i = 0; i < points.size(); i++)
  {
    if (isDumbbellMask[i])
      continue;

    vector < vector<DMatch> > matches;
    bruteForceMatcher.radiusMatch(Mat(points[i]).reshape(1, 1), Mat(points).reshape(1), matches, (float)rBig);

    int pointsInBigCount = static_cast<int> (matches[0].size());

    if (pointsInBigCount <= minPointsInBig)
      continue;

    int pointsInSmallCount = 0;
    for (size_t j = 0; j < matches[0].size(); j++)
    {
      if (matches[0][j].distance < rSmall)
      {
        pointsInSmallCount++;
      }
    }

    if (pointsInSmallCount <= minPointsInSmall)
      continue;

    float ratio = static_cast<float> (pointsInBigCount) / pointsInSmallCount;

    if (ratio > maxDumbellRatio)
      continue;

    for (size_t j = 1; j < matches[0].size(); j++)
    {
      if (matches[0][j].distance < rSmall)
      {
        isDumbbellMask[matches[0][j].trainIdx] = 1;
      }
    }
  }
}

void filterAllDumbbells(std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2,
                        std::vector<uchar> &isDumbbellMask)
{
  CV_Assert(points1.size() == points2.size());
  isDumbbellMask.resize(points1.size(), 0);
  filterDumbbells(points1, isDumbbellMask);
  filterDumbbells(points2, isDumbbellMask);

  vector<Point2f> filteredPoints1, filteredPoints2;
  for (size_t i = 0; i < isDumbbellMask.size(); i++)
  {
    if (!isDumbbellMask[i])
    {
      filteredPoints1.push_back(points1[i]);
      filteredPoints2.push_back(points2[i]);
    }
  }

  std::swap(filteredPoints1, points1);
  std::swap(filteredPoints2, points2);
}

bool HomographyFinder::findInliers(const std::vector<cv::KeyPoint>& keypoints1,
                                   const std::vector<cv::KeyPoint>& keypoints2,
                                   const std::vector<cv::DMatch>& matches1to2, const std::vector<cv::Point2f> &points1,
                                   const std::vector<cv::Point2f> &points2, std::vector<cv::Point2f> &filteredPoints1,
                                   std::vector<cv::Point2f> &filteredPoints2, double distance, cv::Mat &H, vector<
                                       DMatch> &matches1to2Inliers)
{
  CV_Assert(filteredPoints1.size() == filteredPoints2.size());
  if (filteredPoints1.size() < (size_t)minHomographyPoints)
    return false;

  H = findHomography(Mat(filteredPoints1), Mat(filteredPoints2), 0);

  if (isHomographyDegenerate(H))
  {
    return false;
  }

  vector<int> inliers;
  int count = getInliersCount(H, points1, points2, distance, &inliers);
  if (inliers.size() < (size_t)minHomographyPoints)
    return false;

  matches1to2Inliers.clear();
  matches1to2Inliers.reserve(inliers.size());
  for (size_t j = 0; j < inliers.size(); j++)
  {
    matches1to2Inliers.push_back(matches1to2[inliers[j]]);
  }
  orientationConsistencyFilter(keypoints1, keypoints2, matches1to2Inliers, (int)angleDiff);
  getPoints(keypoints1, keypoints2, matches1to2Inliers, filteredPoints1, filteredPoints2);

  if ((size_t)minHomographyPoints < filteredPoints1.size() && filteredPoints1.size() < (size_t)maxInliersCount)
  {
    vector < uchar > isDumbbellMask;
    filterAllDumbbells(filteredPoints1, filteredPoints2, isDumbbellMask);

    vector < DMatch > filteredMatchesInliers;
    for (size_t i = 0; i < isDumbbellMask.size(); i++)
    {
      if (!isDumbbellMask[i])
      {
        filteredMatchesInliers.push_back(matches1to2Inliers[i]);
      }
    }
    std::swap(matches1to2Inliers, filteredMatchesInliers);
  }

  if ((size_t)minHomographyPoints < filteredPoints1.size() && filteredPoints1.size() < (size_t)maxInliersCount)
  {
    CV_Assert(filteredPoints1.size() == filteredPoints2.size());
    geometricConsistency(keypoints1, keypoints2, matches1to2Inliers);
    getPoints(keypoints1, keypoints2, matches1to2Inliers, filteredPoints1, filteredPoints2);
    CV_Assert(filteredPoints1.size() == filteredPoints2.size());
  }

  return true;
}

void HomographyFinder::findHomographyRobustly(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<
    cv::KeyPoint>& keypoints2, const std::vector<cv::DMatch>& matches1to2, cv::Mat &H, int &inliersCount, std::vector<
    cv::DMatch> *matchesInliers)
{
  vector<Point2f> points1, points2;
  getPoints(keypoints1, keypoints2, matches1to2, points1, points2);

  CV_Assert(points1.size() == points2.size());
  inliersCount = 0;

  for (int i = 0; i < findHomographyIterationsCount; i++)
  {
    vector<Point2f> filteredPoints1, filteredPoints2;
    vector<int> indices;
    bool found = getRandomIndices(minHomographyPoints, 0, points1.size(), indices);

    if (!found)
      break;
    for (size_t j = 0; j < indices.size(); j++)
    {
      filteredPoints1.push_back(points1[indices[j]]);
      filteredPoints2.push_back(points2[indices[j]]);
    }

    Mat curH;
    vector < DMatch > matches1to2Inliers;
    if (!findInliers(keypoints1, keypoints2, matches1to2, points1, points2, filteredPoints1, filteredPoints2,
                     ransacReprojectionError, curH, matches1to2Inliers))
      continue;
    if (!findInliers(keypoints1, keypoints2, matches1to2, points1, points2, filteredPoints1, filteredPoints2,
                     lmInlierDistance, curH, matches1to2Inliers))
      continue;
    if (!findInliers(keypoints1, keypoints2, matches1to2, points1, points2, filteredPoints1, filteredPoints2,
                     inlierDistance, curH, matches1to2Inliers))
      continue;

    CV_Assert(filteredPoints1.size() == filteredPoints2.size());
    CV_Assert(matches1to2Inliers.size() == filteredPoints1.size());

    size_t count = filteredPoints1.size();
    if (count > (size_t)inliersCount)
    {
      inliersCount = count;
      H = curH;
      if (matchesInliers != 0)
      {
        std::swap(matches1to2Inliers, *matchesInliers);
      }
      if (inliersCount >= maxInliersCount)
        break;
    }
  }
}
