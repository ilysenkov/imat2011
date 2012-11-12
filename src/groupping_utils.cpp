#pragma warning (disable : 4996)

#include "interfaces.hpp"

using namespace cv;
using namespace std;

KruskalGrouper::KruskalGrouper(double _maxEdgeWeight, int _minPanoSize)
{
  maxEdgeWeight = _maxEdgeWeight;
  minPanoSize = _minPanoSize;
}

KruskalGrouper::Grouping::Grouping(double _threshold, const std::vector<int> &_classes)
{
  threshold = _threshold;
  classes = _classes;
}

int KruskalGrouper::getMaxClassSize(const std::vector<int> &classes)
{
  CV_Assert(classes.size() == images_count);
  vector<int> classes_hist(images_count, 0);
  for (size_t i = 0; i < images_count; ++i)
    classes_hist[classes[i]]++;

  int max_class_idx = std::max_element(classes_hist.begin(), classes_hist.end()) - classes_hist.begin();
  return classes_hist[max_class_idx];
}

void KruskalGrouper::group(const Mat &adjacencyMatrix, std::vector<Grouping> &groupings)
{
  CV_Assert(adjacencyMatrix.type() == CV_64FC1 || adjacencyMatrix.type() == CV_32FC1);
  CV_Assert(adjacencyMatrix.rows == images_count && adjacencyMatrix.cols == images_count);

  const double minValidWeight = 0.0;
  checkRange(adjacencyMatrix, false, 0, minValidWeight);

  Mat classes(images_count, 1, CV_8UC1);
  for (int i = 0; i < images_count; i++)
  {
    classes.at<uchar> (i) = i;
  }

  Mat adjMat;
  adjacencyMatrix.convertTo(adjMat, CV_64FC1);
  for (int i = 0; i < images_count; i++)
  {
    adjMat.at<double> (i, i) = std::numeric_limits<double>::max();
  }

  groupings.clear();
  groupings.push_back(Grouping(minValidWeight, classes.clone()));

  //plus one for the first sentinel
  while (groupings.size() != (images_count - 1) + 1)
  {
    double minVal;
    Point minLoc;
    minMaxLoc(adjMat, &minVal, 0, &minLoc, 0);
    adjMat.at<double> (minLoc.x, minLoc.y) = std::numeric_limits<double>::max();
    adjMat.at<double> (minLoc.y, minLoc.x) = std::numeric_limits<double>::max();

    uchar minLabel = std::min(classes.at<uchar> (minLoc.x), classes.at<uchar> (minLoc.y));
    uchar maxLabel = std::max(classes.at<uchar> (minLoc.x), classes.at<uchar> (minLoc.y));
    if (classes.at<uchar> (minLoc.x) != classes.at<uchar> (minLoc.y))
    {
      classes.setTo(minLabel, classes == maxLabel);
      groupings.push_back(Grouping(minVal, classes.clone()));
    }
  }
  groupings.push_back(Grouping(std::numeric_limits<double>::max(), classes.clone()));
}

void KruskalGrouper::group(const std::vector<Grouping> &groupings, double threshold, int minSize,
                           std::vector<int>& classes)
{
  for (size_t i = 1; i < groupings.size(); i++)
  {
    if (threshold < groupings[i].threshold)
    {
      if (getMaxClassSize(groupings[i - 1].classes) >= minSize)
      {
        classes = groupings[i - 1].classes;
        break;
      }
    }
  }
  CV_Assert(classes.size() == images_count);
}

bool generateOutputMaskFromClasses(vector<int> classes, vector<int>& answer_mask, ostream& console)
{
  CV_Assert(classes.size() == images_count);
  bool found = false;

  /* convert classes to answers_maks */
  answer_mask.clear();
  answer_mask.resize(images_count, 0);

  vector<int> classes_hist(images_count, 0);
  for (size_t i = 0; i < images_count; ++i)
    classes_hist[classes[i]]++;

  int max_class_idx = std::max_element(classes_hist.begin(), classes_hist.end()) - classes_hist.begin();
  int max_class_power = classes_hist[max_class_idx];

  if (max_class_power == 1 || max_class_power == 2)
  {
    found = false;
  }
  else
  {
    for (size_t i = 0; i < classes.size(); ++i)
      if (classes[i] != max_class_idx)
        answer_mask[i] = 1;
    found = true;
  }
  return found;

}
