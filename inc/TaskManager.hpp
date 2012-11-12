#ifndef TASKMANAGER_HPP_HJI234NYHIU3NHEWMJKBNUKE
#define TASKMANAGER_HPP_HJI234NYHIU3NHEWMJKBNUKE

#include "interfaces.hpp"

struct TaskManager
{
public:
  TaskManager(const std::string& folder, int from = 1001, int to = 6000);

  void run(std::string groundTruthFile = "");

  int from;
  int to;

  Answers::type answers;

  std::string cache_folder;
private:
  std::string folder;
};

#endif /* TaskManager_HPP_HJI234NYHIU3NHEWMJKBNUKE */
