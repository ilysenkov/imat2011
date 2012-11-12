#ifndef HOMOGRAPHYFINDER_HPP_
#define HOMOGRAPHYFINDER_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

void orientationConsistencyFilter(const std::vector<cv::KeyPoint>& keypoints1,
                                  const std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& matches1to2,
                                  int angleDiff = 5);

class HomographyFinder
{
public:
  HomographyFinder();

  void findHomographyRobustly(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
                              const std::vector<cv::DMatch>& matches1to2, cv::Mat &H, int &inliersCount, std::vector<
                                  cv::DMatch> *matchesInliers = 0);
  bool isHomographyDegenerate(const cv::Mat &H);
  int getInliersCount(const cv::Mat &H, const std::vector<cv::Point2f> &points1,
                      const std::vector<cv::Point2f> &points2, double distance, std::vector<int> *inliers = 0);
private:
  bool findInliers(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
                   const std::vector<cv::DMatch>& matches1to2, const std::vector<cv::Point2f> &points1,
                   const std::vector<cv::Point2f> &points2, std::vector<cv::Point2f> &filteredPoints1, std::vector<
                       cv::Point2f> &filteredPoints2, double distance, cv::Mat &H,
                   std::vector<cv::DMatch> &matches1to2Inliers);

  double minSingularValue;
  double lmInlierDistance;
  double inlierDistance;
  double ransacReprojectionError;
  int findHomographyIterationsCount;
  double angleDiff;

  int minHomographyPoints;

  int maxInliersCount;
};

#endif /* HOMOGRAPHYFINDER_HPP_ */
