#pragma warning (disable : 4996)

#include "interfaces.hpp"
#include "lexical_cast.hpp"

using namespace cv;

HistogramStrategy::HistogramStrategy(const std::string& cache_folder) :
  MatchingStrategy(cache_folder)
{
  absoluteConfidence = 51000;
  randomConfidence = 59300;
  zeroConfidence = 90000;
}

void HistogramStrategy::launch(const std::vector<cv::Mat>& images, int seria_ind)
{
  const int lumaBins = 12, crBins = 256, cbBins = 256;
  const int yBins = 4;

  checkFlagAndLoadFromCache(seria_ind);

  if (!use_cache || differences.empty())
  {
    const int dims = 4;
    const int imageSize = 300;
    Mat yChannel(imageSize, imageSize, CV_8UC1);
    for (int i = 0; i < imageSize; i++)
    {
      yChannel.row(i).setTo(i);
    }

    int histSize[] = {lumaBins, crBins, cbBins, yBins};
    float lumaRanges[] = {0, 256};
    float crRanges[] = {0, 256};
    float cbRanges[] = {0, 256};

    //This was fixed after the competition
    //float yRanges[] = {0, 256};
    float yRanges[] = {0, 300};
    
    const float* ranges[] = {lumaRanges, crRanges, cbRanges, yRanges};
    vector < MatND > hist(images_count);
    int channels[] = {0, 1, 2, 3};

    for (size_t imIndex = 0; imIndex < images.size(); imIndex++)
    {
      if (!images[imIndex].empty())
      {
        Mat ycrcb;
        cvtColor(images[imIndex], ycrcb, CV_BGR2YCrCb);

        vector < Mat > imagesForCalcHist;
        imagesForCalcHist.push_back(ycrcb);
        imagesForCalcHist.push_back(yChannel);
        calcHist(&imagesForCalcHist[0], 2, channels, Mat(), // do not use mask
                 hist[imIndex], dims, histSize, ranges, true, // the histogram is uniform
                 false);
      }
      else
      {
        hist[imIndex].create(crBins, cbBins, CV_32F);
        hist[imIndex].setTo(0.f);
      }
    }

    differences.create(images.size(), images.size(), CV_32F);
    differences.setTo(Scalar::all(0));

    const int pixelsCount = imageSize * imageSize;
    for (size_t i = 0; i < images.size(); i++)
    {
      for (size_t j = 0; j < i; j++)
      {
        double dist = pixelsCount - compareHist(hist[i], hist[j], CV_COMP_INTERSECT);
        differences.at<float> (i, j) = differences.at<float> (j, i) = static_cast<float> (dist);
      }
    }
  }

  checkFlagAndSaveCache(seria_ind);
}

void HistogramStrategy::checkFlagAndSaveCache(int seria_ind) const
{
  if (gen_cache && !cache_folder.empty())
  {
    if (differences.empty() && adjacencyMatrix.empty())
      return;

    cv::FileStorage fs(cache_folder + "/hist_cache_" + lexical_cast<string> (seria_ind) + ".xml", FileStorage::WRITE);

    if (!differences.empty())
      fs << "differences" << differences;

    if (!adjacencyMatrix.empty())
      fs << "adjacencyMatrix" << adjacencyMatrix;
  }
}

void HistogramStrategy::checkFlagAndLoadFromCache(int seria_ind)
{
  if (use_cache && !cache_folder.empty())
  {
    cv::FileStorage fs(cache_folder + "/hist_cache_" + lexical_cast<string> (seria_ind) + ".xml", FileStorage::READ);
    fs["differences"] >> differences;
    fs["adjacencyMatrix"] >> adjacencyMatrix;
  }
}
