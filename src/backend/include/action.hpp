#include "events.hpp"
#include "settings.hpp"
#include <cstddef>

class Action {
public:
  taskid_t task_id;
  std::size_t pos;
  devid_t device;
  priority_t reservable_priority;
  priority_t launchable_priority;
};

using ActionList = std::vector<Action>;