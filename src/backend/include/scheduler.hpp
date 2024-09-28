#pragma once
#include "queues.hpp"
#include "settings.hpp"

using TaskQueue = ContainerQueue<taskid_t, std::priority_queue>;
using DeviceQueue = std::vector<TaskQueue>;
