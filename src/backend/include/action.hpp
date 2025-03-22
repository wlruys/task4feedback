#pragma once
#include "events.hpp"
#include "settings.hpp"
#include <cstddef>

class Action {
public:
  std::size_t pos = 0;
  int32_t device = 0;
  int32_t reservable_priority = 0;
  int32_t launchable_priority = 0;
};

using ActionList = std::vector<Action>;