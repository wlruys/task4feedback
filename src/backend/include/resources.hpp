#pragma once
#include "settings.hpp"
#include <array>

enum class ResourceType { VCUS = 0, MEM = 1, TIME = 2 };
constexpr std::size_t num_resource_types = 3;

using vcu_t = unsigned int;
using mem_t = unsigned long long;
using timecount_t = unsigned long long;
using copy_t = unsigned int;

constexpr vcu_t MAX_VCUS = 1000;
constexpr mem_t BYTES_IN_POWER = 1024;

consteval mem_t operator"" _KB(mem_t val) { return val * BYTES_IN_POWER; }
consteval mem_t operator"" _MB(mem_t val) {
  return val * BYTES_IN_POWER * BYTES_IN_POWER;
}
consteval mem_t operator"" _GB(mem_t val) {
  return val * BYTES_IN_POWER * BYTES_IN_POWER * BYTES_IN_POWER;
}

consteval timecount_t operator"" _us(unsigned long long val) { return val; }
consteval timecount_t operator"" _ms(unsigned long long val) {
  return val * 1000;
}
consteval timecount_t operator"" _s(unsigned long long val) {
  return val * 1000 * 1000;
}

consteval vcu_t operator"" _vcus(long double val) {
  // Return fraction of MAX_VCUS
  return static_cast<vcu_t>(val * MAX_VCUS);
}