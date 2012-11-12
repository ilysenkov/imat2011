#pragma warning (disable : 4996)

#include "lexical_cast.hpp"
#include "TaskManager.hpp"

using std::string;

int main(int argc, char **argv)
{
  cv::setBreakOnError(true);

  if(argc < 4)
  {
    std::cout << argv[0] << " <from> <to> <images_folder> [cache_folder]" << std::endl;
    return 0;
  }

  string images_folder = argv[3];

  TaskManager tasks(images_folder);
  tasks.cache_folder = (argc == 5) ? argv[4] : "";
  tasks.from = lexical_cast<int> (string(argv[1]));
  tasks.to = lexical_cast<int> (string(argv[2]));

  tasks.run();

  Answers::saveAnswers("tosend.txt", tasks.answers);
  return 0;
}
