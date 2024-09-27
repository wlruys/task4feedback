#pragma once

#include "macros.hpp"
#include "queues.hpp"
#include "settings.hpp"
#include "tasks.hpp"
#include <queue>

using TaskQueue = Randomizer<taskid_t, std::priority_queue>;
using DeviceQueue = std::vector<TaskQueue>;
