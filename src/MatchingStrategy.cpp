#pragma warning (disable : 4996)

#include "interfaces.hpp"

using namespace std;
using namespace cv;

float MatchingStrategy::normalizeDistance(float distance) const
{
  if (distance < absoluteConfidence)
    return 0.0f;

  if (distance < randomConfidence)
    return (float)(0.5 * normalize(distance, absoluteConfidence, randomConfidence));

  if (distance < zeroConfidence)
    return (float)(0.5 + 0.5 * normalize(distance, randomConfidence, zeroConfidence));

  return 1.0f;
}

void MatchingStrategy::normalizeDifferences(const cv::Mat &differences, cv::Mat &result) const
{
  CV_Assert(differences.rows == images_count);
  CV_Assert(differences.cols == images_count);
  CV_Assert(differences.type() == CV_32FC1);

  result.create(differences.size(), differences.type());
  result.setTo(Scalar(0));
  for (int i = 0; i < differences.rows; i++)
  {
    for (int j = 0; j < differences.cols; j++)
    {
      if (i == j)
        continue;

      float dist = normalizeDistance(differences.at<float> (i, j));
      result.at<float> (i, j) = dist;
      //result.at<float>(i, j) = result.at<float>(j, i) = dist;
    }
  }
}
