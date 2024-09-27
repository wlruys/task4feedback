#pragma once

#include "devices.hpp"
#include "resources.hpp"

enum class DataState {
  NONE = 0,
  PLANNED = 1,
  MOVING = 2,
  VALID = 3,
};
constexpr std::size_t num_data_states = 4;
