#pragma once
#include "settings.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <sys/types.h>

enum class ResourceType { VCUS = 0, MEM = 1, TIME = 2 };
constexpr std::size_t num_resource_types = 3;

using vcu_t = uint16_t;
using mem_t = uint64_t;
using timecount_t = uint64_t;
using copy_t = uint8_t;

constexpr mem_t BYTES_IN_POWER = 1024;

constexpr vcu_t MAX_VCUS = 1000;
constexpr mem_t MAX_MEM = std::numeric_limits<mem_t>::max();
constexpr timecount_t MAX_TIME = std::numeric_limits<timecount_t>::max();
constexpr copy_t MAX_COPIES = 10;

consteval mem_t operator"" _KB(mem_t val) { return val * BYTES_IN_POWER; }
consteval mem_t operator"" _MB(mem_t val) {
  return val * BYTES_IN_POWER * BYTES_IN_POWER;
}
consteval mem_t operator"" _GB(mem_t val) {
  return val * BYTES_IN_POWER * BYTES_IN_POWER * BYTES_IN_POWER;
}

consteval timecount_t operator"" _us(timecount_t val) { return val; }
consteval timecount_t operator"" _ms(timecount_t val) { return val * 1000; }
consteval timecount_t operator"" _s(timecount_t val) {
  return val * 1000 * 1000;
}

consteval vcu_t operator"" _vcus(long double val) {
  // Return fraction of MAX_VCUS
  return static_cast<vcu_t>(val * MAX_VCUS);
}