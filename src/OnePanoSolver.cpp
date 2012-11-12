#pragma warning (disable :4996)

#include "interfaces.hpp"
#include "lexical_cast.hpp"

#include <sstream>

using namespace std;
using namespace cv;

void OnePanoSolver::readImages(const std::string& folder, int seria_index, std::vector<cv::Mat> &imgs)
{
  string ss = lexical_cast<string> (seria_index);

  imgs.resize(5);
  imgs[0] = imread(folder + "/" + ss + "_1.jpg");
  imgs[1] = imread(folder + "/" + ss + "_2.jpg");
  imgs[2] = imread(folder + "/" + ss + "_3.jpg");
  imgs[3] = imread(folder + "/" + ss + "_4.jpg");
  imgs[4] = imread(folder + "/" + ss + "_5.jpg");
}

OnePanoSolver::OnePanoSolver(const std::string& folder, int ser_index, const std::string& cache_folder__)
{
  cache_folder = cache_folder__;
  seria_index = ser_index;

  readImages(folder, seria_index, images);
}

void OnePanoSolver::launch(std::vector<KruskalGrouper::Grouping> &groupings, Mat& diff)
{
  Ptr<HistogramStrategy> hist_strategy = new HistogramStrategy(cache_folder);
  hist_strategy->launch(images, seria_index);

  Ptr<CrossCheckStrategy> cross_check_strategy = new CrossCheckStrategy(cache_folder);
  cross_check_strategy->launch(images, seria_index);

  Mat diff_hist, diff_cross_check;
  hist_strategy->normalizeDifferences(hist_strategy->differences, diff_hist);
  cross_check_strategy->normalizeDifferences(cross_check_strategy->adjacencyMatrix, diff_cross_check);

  const float hist_dumping_factor = 0.66f;
  diff = min(hist_dumping_factor * diff_hist, diff_cross_check);

  KruskalGrouper kruskalGrouper;
  kruskalGrouper.group(diff, groupings);

  cout << diff << endl;
}

void OnePanoSolver::launch(std::vector<KruskalGrouper::Grouping> &groupings, Mat& diff, int)
{
  Ptr<CrossCheckStrategy> strategy = new CrossCheckStrategy(cache_folder);
  strategy->launch(images, seria_index);

  diff = strategy->adjacencyMatrix.clone();

  KruskalGrouper kruskalGrouper;
  kruskalGrouper.group(strategy->adjacencyMatrix, groupings);
}
