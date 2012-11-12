#pragma warning (disable : 4996)

#include <fstream>
#include <iterator>
#include <iomanip>
#include "lexical_cast.hpp"
#include "interfaces.hpp"

using namespace std;
using namespace cv;

float normalize(float x, float a, float b)
{
  CV_Assert(a <= x);
  CV_Assert(x <= b);
  return (x - a) / (b - a);
}

void Answers::loadAnswers(const string& file, map<int, vector<int> >& out)
{
  out.clear();
  ifstream ifs(file.c_str());

  if (!ifs)
    throw "bad";

  istream_iterator < string > pos(ifs);
  istream_iterator < string > end;

  for (int ser_num, in_ser_num; pos != end; ++pos)
  {
    sscanf(pos->c_str(), "%d_%d.jpg", &ser_num, &in_ser_num);
    vector<int>& masks = out[ser_num];

    if (masks.size() != 5)
    {
      masks.clear();
      masks.resize(5, 0);
    }
    masks[in_ser_num - 1] = 1;
  }
}

void Answers::saveAnswers(const string& file, const map<int, vector<int> >& in)
{
  vector < string > list;
  typedef map<int, vector<int> >::const_iterator It;
  for (It pos = in.begin(); pos != in.end(); ++pos)
    for (size_t i = 0; i < pos->second.size(); ++i)
      if (pos->second[i])
        list.push_back(lexical_cast<string> (pos->first) + "_" + lexical_cast<string> (i + 1) + ".jpg");

  ofstream ofs(file.c_str());
  if (!ofs)
    throw "bad";
  ostream_iterator < string > out(ofs, "\n\n");
  copy(list.begin(), list.end(), out);
}
