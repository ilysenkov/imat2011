#pragma warning (disable : 4996)

#include "TaskManager.hpp"
#include "lexical_cast.hpp"

#include <omp.h>

using namespace std;
using namespace cv;

TaskManager::TaskManager(const std::string& fld, int f, int t) :
  folder(fld), from(f), to(t)
{
}

void TaskManager::run(string groundTruthFile)
{
  //run on the train data or on the test data
  bool useGroundTruth = !groundTruthFile.empty();
  cout << "Processing from " << from << " to " << to << endl << endl;
  answers.clear();

  Answers::type rightAnswers;
  if (useGroundTruth)
  {
    Answers::loadAnswers(groundTruthFile, rightAnswers);
    const int inlierLabel = 0;
    vector<int> fullPanoramaMask(images_count, inlierLabel);

    for (int i = from; i <= to; ++i)
      if (rightAnswers.find(i) == rightAnswers.end())
        rightAnswers.insert(make_pair(i, fullPanoramaMask));
  }

  int total = to - from + 1;
  vector < vector<KruskalGrouper::Grouping> > groupings(total);

  float minIncorrectDistance = std::numeric_limits<float>::max();
  int minIncorrectDistanceSeriaIdx = -1;

#pragma omp parallel for schedule(dynamic, 5)
  for (int i = from; i <= to; ++i)
  {
    cout << "Seria #" << i << "\t" << endl;
    TickMeter time;
    time.start();
    OnePanoSolver solver(folder, i, cache_folder);
    Mat diff;

#if 0 // cross-check only
    bool found = solver.launch(groupings[i - from], diff, int());
#else
    solver.launch(groupings[i - from], diff);
#endif

    time.stop();
    cout << "Time: " << time.getTimeSec() << "s" << endl;

    if (useGroundTruth)
    {
      vector<int> right = rightAnswers[i];
      for (int j = 0; j < diff.rows; j++)
      {
        for (int k = j + 1; k < diff.cols; k++)
        {
          if (right[j] != right[k])
          {
            CV_Assert(diff.type() == CV_32FC1);
            if (diff.at<float> (j, k) <= minIncorrectDistance)
            {
              minIncorrectDistance = diff.at<float> (j, k);
              minIncorrectDistanceSeriaIdx = i;
            }
          }
        }
      }
    }
  }

  int bestScore = -1;
  double bestThreshold = -1;
  const int minPanoSize = 3;

  if (useGroundTruth)
  {
    for (size_t i = 0; i < groupings.size(); i++)
    {
      int curBestScore = -1;
      double curBestThreshold = -1;

      for (size_t j = 1; j < groupings[i].size() - 1; j++)
      {
        double curThreshold = groupings[i][j].threshold;
        int curScore = 0;
        for (size_t k = 0; k < groupings.size(); k++)
        {
          vector<int> classes, answer_mask;
          KruskalGrouper::group(groupings[k], curThreshold, minPanoSize, classes);
          stringstream devNull;
          generateOutputMaskFromClasses(classes, answer_mask, devNull);

          int score = static_cast<int>(images_count - norm(Mat(answer_mask) - Mat(rightAnswers[from + k]), NORM_L1));
          curScore += score;
        }

        if (curScore > curBestScore)
        {
          curBestScore = curScore;
          curBestThreshold = curThreshold;
        }
      }
      if (curBestScore > bestScore)
      {
        bestScore = curBestScore;
        bestThreshold = curBestThreshold;
      }
    }
  }
  else
  {
    bestThreshold = 0.3;
  }

  for (size_t k = 0; k < groupings.size(); k++)
  {
    vector<int> classes;
    KruskalGrouper::group(groupings[k], bestThreshold, minPanoSize, classes);

    stringstream devNull;
    generateOutputMaskFromClasses(classes, answers[k + from], devNull);
  }
}
