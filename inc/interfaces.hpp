#ifndef PROBLEM_SOLVER_HPP_KLDAHJICVUHQ34YT8ORIUFM32HUNV98BRGNF8I23OW
#define PROBLEM_SOLVER_HPP_KLDAHJICVUHQ34YT8ORIUFM32HUNV98BRGNF8I23OW

#include<vector>
#include<map>
#include<string>
#include<sstream>
#include<opencv2/opencv.hpp>

/*****************************************************/
/****************** Constants ************************/
/*****************************************************/

const int images_count = 5;

/*****************************************************/
/******************** Types **************************/
/*****************************************************/

typedef std::vector<int> mask_type;

struct Answers
{
  typedef std::map<int, std::vector<int> > type;

  static void loadAnswers(const std::string& file, type& out);
  static void saveAnswers(const std::string& file, const type& in);
};

/*****************************************************/
/****************** Functions ************************/
/*****************************************************/
float normalize(float x, float a, float b);

class KruskalGrouper
{
public:
  struct Grouping
  {
    double threshold;
    std::vector<int> classes;
    int maxClassLabel;

    Grouping(double threshold, const std::vector<int> &classes);
  };

  KruskalGrouper(double maxEdgeWeight = -1, int minPanoSize = 0);

  static void group(const std::vector<Grouping> &groupings, double threshold, int minSize, std::vector<int>& classes);
  static void group(const cv::Mat &adjacencyMatrix, std::vector<Grouping> &groupings);
  static int getMaxClassSize(const std::vector<int> &classes);
private:

  double maxEdgeWeight;
  int minPanoSize;
};

bool generateOutputMaskFromClasses(std::vector<int> classes, std::vector<int>& answer_mask, std::ostream& console);

/*****************************************************/
/****************** Strategies ***********************/
/*****************************************************/
/****************** Base interface *******************/
class MatchingStrategy
{
public:
  MatchingStrategy(const std::string& cache_folder_) :
    cache_folder(cache_folder_)
  {
  }

  virtual void launch(const std::vector<cv::Mat>& images, int seria_ind) = 0;

  virtual float normalizeDistance(float distance) const;
  virtual void normalizeDifferences(const cv::Mat &differences, cv::Mat &normalizedDifferences) const;

  virtual ~MatchingStrategy()
  {
  }

  cv::Mat adjacencyMatrix;
protected:
  std::string cache_folder;
  float absoluteConfidence, randomConfidence, zeroConfidence;
};

/*****************************************************/
/****************** HISTOGRAM ************************/
class HistogramStrategy : public MatchingStrategy
{
public:
  HistogramStrategy(const std::string& cache_folder = std::string());

  virtual void launch(const std::vector<cv::Mat>& images, int seria_ind);

  cv::Mat differences;
  const static int use_cache = 0;
  const static int gen_cache = 0;
private:
  void checkFlagAndSaveCache(int seria_ind) const;
  void checkFlagAndLoadFromCache(int seria_ind);
};

/***********************************************************/
/****************** CrossCheckStrategy *********************/
class CrossCheckStrategy : public MatchingStrategy
{
public:
  CrossCheckStrategy(const std::string &cache_folder);

  virtual void launch(const std::vector<cv::Mat>& images, int seria_ind);

  const static int use_cache = 0;
  const static int gen_cache = 0;

  const static int use_flann_inliers_cache = 0;
  const static int gen_flann_inliers_cache = 0;

private:
  const static int flann_search_count = 128;

  cv::Ptr<cv::FeatureDetector> featureDetector;
  cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;

  void checkCacheAndSave(int seria_ind, int y, int x, const std::vector<cv::KeyPoint>& keypointsY, const std::vector<
      cv::KeyPoint>& keypointsX, const std::vector<cv::DMatch>& matchesYtoX) const;

  bool checkCacheAndLoad(int seriad_ind, int y, int x, std::vector<cv::KeyPoint>& keypointsY,
                         std::vector<cv::KeyPoint>& keypointsX, std::vector<cv::DMatch>& matchesYtoX);
  void loadInliers(int seria_ind, const std::vector<cv::Mat>& images,
                   std::vector<std::vector<cv::KeyPoint> >& keypointsY,
                   std::vector<std::vector<cv::KeyPoint> >& keypointsX,
                   std::vector<std::vector<cv::DMatch> >& matchInliers);
  void calcInliers(const std::vector<cv::Mat>& images, std::vector<std::vector<cv::KeyPoint> >& keypointsY,
                   std::vector<std::vector<cv::KeyPoint> >& keypointsX,
                   std::vector<std::vector<cv::DMatch> >& matchInliers);

  void checkFlagAndSaveCache(int seria_ind) const;
  void checkFlagAndLoadFromCache(int seria_ind);
};

/*****************************************************/
/***************** Single Solver *********************/
/*****************************************************/

class OnePanoSolver
{
public:
  static void readImages(const std::string& folder, int seria_index, std::vector<cv::Mat> &images);
  OnePanoSolver(const std::string& folder, int seria_index, const std::string& cache_folder = std::string());

  virtual void launch(std::vector<KruskalGrouper::Grouping> &groupings, cv::Mat& diff);
  virtual void launch(std::vector<KruskalGrouper::Grouping> &groupings, cv::Mat& diff, int);

protected:
  cv::Mat adjacencyMatrix;
  int seria_index;
  std::vector<cv::Mat> images;
  std::vector<int> classes;
  bool found;
  std::string cache_folder;
};

#endif

