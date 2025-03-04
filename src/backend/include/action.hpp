#pragma once
#include "events.hpp"
#include "settings.hpp"
#include <cstddef>

class Action {
public:
  std::size_t pos = 0;
  devid_t device = 0;
  priority_t reservable_priority = 0;
  priority_t launchable_priority = 0;
};

using ActionList = std::vector<Action>;