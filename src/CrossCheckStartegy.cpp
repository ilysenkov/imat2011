#pragma warning (disable : 4996)

#include "interfaces.hpp"
#include "opencv2/flann/flann.hpp"
#include "lexical_cast.hpp"
#include "homographyFinder.hpp"

using namespace std;
using namespace cv;

#define GET_ORIENTATION 1

CrossCheckStrategy::CrossCheckStrategy(const std::string &cache_folder) :
  MatchingStrategy(cache_folder)
{
  absoluteConfidence = 99979;
  randomConfidence = 99988;
  zeroConfidence = 100000;

  featureDetector = new PyramidAdaptedFeatureDetector(new FastFeatureDetector(10, false), 3);
  descriptorExtractor = new SurfDescriptorExtractor();
}

void crossCheckMatching(Ptr<DescriptorMatcher>& descriptorMatcherTo1, Ptr<DescriptorMatcher>& descriptorMatcherTo2,
                        const Mat& descriptors1, const Mat& descriptors2, vector<DMatch>& filteredMatches12)
{
  filteredMatches12.clear();
  vector<DMatch> matches12, matches21;
  descriptorMatcherTo2->match(descriptors1, matches12);
  descriptorMatcherTo1->match(descriptors2, matches21);
  for (size_t m = 0; m < matches12.size(); m++)
  {
    DMatch forward = matches12[m];
    DMatch backward = matches21[forward.trainIdx];
    if (backward.trainIdx == forward.queryIdx)
    {
      filteredMatches12.push_back(forward);
    }
  }
}

void filterMatchesByY(const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2,
                      std::vector<cv::DMatch> &matches)
{
  const float maxDy = 100;
  vector < uchar > isOk(matches.size(), 1);
  for (size_t i = 0; i < matches.size(); i++)
  {
    Point2f pt1 = keypoints1.at(matches[i].queryIdx).pt;
    Point2f pt2 = keypoints2.at(matches[i].trainIdx).pt;
    float dy = fabs(pt1.y - pt2.y);
    if (dy > maxDy)
    {
      isOk[i] = 0;
    }
  }

  vector < DMatch > filteredMatches;
  for (size_t i = 0; i < isOk.size(); i++)
  {
    if (isOk[i])
    {
      filteredMatches.push_back(matches[i]);
    }
  }

  std::swap(matches, filteredMatches);
}

inline int getIndex(int y, int x, int seriaSize)
{
  return y * seriaSize + x - ((y + 2) * (y + 1)) / 2;
}

void CrossCheckStrategy::loadInliers(int seria_ind, const std::vector<cv::Mat>& images,
                                     vector<vector<KeyPoint> >& keypointsY, vector<vector<KeyPoint> >& keypointsX,
                                     vector<vector<DMatch> >& matchInliers)
{
  CV_Assert(!use_cache || adjacencyMatrix.empty());
  CV_Assert(use_flann_inliers_cache);

  const int seriaSize = images.size();
  CV_Assert(seriaSize == images_count);

  vector < vector<KeyPoint> > keypoints(seriaSize);
  vector < Mat > descriptors(seriaSize);
  vector < Ptr<DescriptorMatcher> > descMatchers(seriaSize);

  keypointsY.resize(10);
  keypointsX.resize(10);
  matchInliers.resize(10);

  cvflann::set_distance_type(cvflann::FLANN_DIST_L1, 1);

  for (int y = 0; y < seriaSize; y++)
  {
    for (int x = y + 1; x < seriaSize; x++)
    {
      int index = getIndex(y, x, seriaSize);

      bool loaded = CrossCheckStrategy::checkCacheAndLoad(seria_ind, y, x, keypointsY[index], keypointsX[index],
                                                          matchInliers[index]);
      if (!loaded)
      {
        // compute data for y-image
        if (descMatchers[y].empty())
        {
          CV_Assert(keypoints[y].empty());
          featureDetector->detect(images[y], keypoints[y]);
#if GET_ORIENTATION
          // hack for opencv to get keypoints orientations
          keypoints[y].push_back(KeyPoint(Point2f(-1000, -1000), 1));
#endif
          CV_Assert(descriptors[y].empty());
          descriptorExtractor->compute(images[y], keypoints[y], descriptors[y]);

          descMatchers[y] = new FlannBasedMatcher(new flann::KDTreeIndexParams(),
                                                  new flann::SearchParams(flann_search_count));
          descMatchers[y]->add(vector<Mat> (1, descriptors[y]));
        }

        // compute data for x-image
        if (descMatchers[x].empty())
        {
          CV_Assert(keypoints[x].empty());
          featureDetector->detect(images[x], keypoints[x]);
#if GET_ORIENTATION
          // hack for opencv to get keypoints orientations
          keypoints[x].push_back(KeyPoint(Point2f(-1000, -1000), 1));
#endif
          CV_Assert(descriptors[x].empty());
          descriptorExtractor->compute(images[x], keypoints[x], descriptors[x]);

          descMatchers[x] = new FlannBasedMatcher(new flann::KDTreeIndexParams(),
                                                  new flann::SearchParams(flann_search_count));
          descMatchers[x]->add(vector<Mat> (1, descriptors[x]));
        }

        if (descriptors[y].empty() || descriptors[x].empty())
        {
          continue;
        }

        int index = getIndex(y, x, seriaSize);

        keypointsY[index] = keypoints[y];
        keypointsX[index] = keypoints[x];
        crossCheckMatching(descMatchers[y], descMatchers[x], descriptors[y], descriptors[x], matchInliers[index]);
      }
    }
  }
}

void CrossCheckStrategy::calcInliers(const std::vector<cv::Mat>& images, vector<vector<KeyPoint> >& keypointsY, vector<
    vector<KeyPoint> >& keypointsX, vector<vector<DMatch> >& matchInliers)
{
  const int seriaSize = images.size();
  CV_Assert(seriaSize == images_count);

  vector < vector<KeyPoint> > keypoints(seriaSize);
  vector < Mat > descriptors(seriaSize);
  vector < Ptr<DescriptorMatcher> > descMatchers(seriaSize);

  keypointsY.resize(10);
  keypointsX.resize(10);
  matchInliers.resize(10);
  cvflann::set_distance_type(cvflann::FLANN_DIST_L1, 1);

  for (size_t i = 0; i < images.size(); i++)
  {
    CV_Assert(!images[i].empty());
    {
      featureDetector->detect(images[i], keypoints[i]);
#if GET_ORIENTATION
      keypoints[i].push_back(KeyPoint(Point2f(-1000, -1000), 1));
#endif
      descriptorExtractor->compute(images[i], keypoints[i], descriptors[i]);

      descMatchers[i] = new FlannBasedMatcher(new flann::KDTreeIndexParams(),
                                              new flann::SearchParams(flann_search_count));
      descMatchers[i]->add(vector<Mat> (1, descriptors[i]));
    }
  }

  for (int y = 0; y < seriaSize; y++)
  {
    for (int x = y + 1; x < seriaSize; x++)
    {
      if (descriptors[y].empty() || descriptors[x].empty())
      {
        continue;
      }

      int index = getIndex(y, x, seriaSize);
      keypointsY[index] = keypoints[y];
      keypointsX[index] = keypoints[x];
      crossCheckMatching(descMatchers[y], descMatchers[x], descriptors[y], descriptors[x], matchInliers[index]);
    }
  }
}

void CrossCheckStrategy::launch(const std::vector<cv::Mat>& images, int seria_ind)
{
  checkFlagAndLoadFromCache(seria_ind);

  vector < vector<KeyPoint> > keypointsY;
  vector < vector<KeyPoint> > keypointsX;
  vector < vector<DMatch> > matchInliers;

  if (!use_cache || adjacencyMatrix.empty())
  {
    const int seriaSize = images.size();
    if (use_flann_inliers_cache)
      loadInliers(seria_ind, images, keypointsY, keypointsX, matchInliers);
    else
      calcInliers(images, keypointsY, keypointsX, matchInliers);

    adjacencyMatrix.create(seriaSize, seriaSize, CV_32FC1);
    const double maxInliersCount = 100000;
    adjacencyMatrix = maxInliersCount;
    adjacencyMatrix.diag().setTo(0);

    for (int y = 0; y < seriaSize; y++)
    {
      for (int x = y + 1; x < seriaSize; x++)
      {
        int inliersCount = 0;
        int index = getIndex(y, x, seriaSize);

        if (matchInliers[index].empty())
          continue;

        checkCacheAndSave(seria_ind, y, x, keypointsY[index], keypointsX[index], matchInliers[index]);

        filterMatchesByY(keypointsY[index], keypointsX[index], matchInliers[index]);

        vector < DMatch > matchesYtoXInliers;
        Mat H12;
        {
          HomographyFinder homographyFinder;
          homographyFinder.findHomographyRobustly(keypointsY[index], keypointsX[index], matchInliers[index], H12,
                                                  inliersCount);
        }
        if (inliersCount <= 0)
          continue;

        float dist = static_cast<float>(maxInliersCount - inliersCount);
        adjacencyMatrix.at<float> (y, x) = adjacencyMatrix.at<float> (x, y) = dist;
      }
    }
  }
  checkFlagAndSaveCache(seria_ind);
}

void CrossCheckStrategy::checkCacheAndSave(int seria_ind, int y, int x, const vector<KeyPoint>& keypointsY,
                                           const vector<KeyPoint>& keypointsX, const vector<DMatch>& matchesYtoX) const
{
  if (!cache_folder.empty() && gen_flann_inliers_cache)
  {
    string file = cache_folder + "/flann_inliers_cache" + "_" + lexical_cast<string> (seria_ind) + "_" + lexical_cast<
        string> (y) + "_" + lexical_cast<string> (x) + ".dt";
    FILE *f = fopen(file.c_str(), "w+b");

    size_t total = matchesYtoX.size();
    fwrite(&total, sizeof(total), 1, f);

    for (size_t d = 0; d < matchesYtoX.size(); ++d)
    {
      KeyPoint kpY = keypointsY[matchesYtoX[d].queryIdx];
      KeyPoint kpX = keypointsX[matchesYtoX[d].trainIdx];
      float dist = matchesYtoX[d].distance;

      fwrite(&kpY, sizeof(kpY), 1, f);
      fwrite(&kpX, sizeof(kpX), 1, f);
      fwrite(&dist, sizeof(dist), 1, f);
    }

    fclose(f);
  }
}

bool CrossCheckStrategy::checkCacheAndLoad(int seria_ind, int y, int x, vector<KeyPoint>& keypointsY,
                                           vector<KeyPoint>& keypointsX, vector<DMatch>& matchesYtoX)
{
  if (!cache_folder.empty() && use_flann_inliers_cache)
  {
    string file = cache_folder + "/flann_inliers_cache" + "_" + lexical_cast<string> (seria_ind) + "_" + lexical_cast<
        string> (y) + "_" + lexical_cast<string> (x) + ".dt";

    FILE *f = fopen(file.c_str(), "r+b");
    if (!f)
      return false;

    size_t total;
    fread(&total, sizeof(total), 1, f);

    keypointsY.resize(total);
    keypointsX.resize(total);
    matchesYtoX.resize(total);

    for (size_t k = 0; k < total; ++k)
    {
      KeyPoint kpY;
      KeyPoint kpX;
      float dist;

      size_t c1 = fread(&kpY, sizeof(kpY), 1, f);
      size_t c2 = fread(&kpX, sizeof(kpX), 1, f);
      size_t c3 = fread(&dist, sizeof(dist), 1, f);

      if (c1 == 0 || c2 == 0 || c3 == 0)
        return false;

      keypointsY[k] = kpY;
      keypointsX[k] = kpX;
      matchesYtoX[k] = DMatch(k, k, dist);
    }
    fclose(f);
    return true;
  }
  return false;
}

void CrossCheckStrategy::checkFlagAndSaveCache(int seria_ind) const
{
  if (gen_cache && !cache_folder.empty())
  {
    if (adjacencyMatrix.empty())
      return;

    cv::FileStorage fs(cache_folder + "/cross_check_cache_" + lexical_cast<string> (seria_ind) + ".xml",
                       FileStorage::WRITE);

    if (!adjacencyMatrix.empty())
      fs << "adjacencyMatrix" << adjacencyMatrix;

    fs.release();
  }
}

void CrossCheckStrategy::checkFlagAndLoadFromCache(int seria_ind)
{
  if (use_cache && !cache_folder.empty())
  {
    cv::FileStorage fs(cache_folder + "/cross_check_cache_" + lexical_cast<string> (seria_ind) + ".xml",
                       FileStorage::READ);
    fs["adjacencyMatrix"] >> adjacencyMatrix;
    fs.release();
  }
}
