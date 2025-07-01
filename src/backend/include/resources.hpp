#pragma once
#include "settings.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <sys/types.h>

enum class ResourceType : int8_t {
  VCUS = 0,
  MEM = 1,
  TIME = 2
};
constexpr std::size_t num_resource_types = 3;

constexpr mem_t BYTES_IN_POWER = 1024;

constexpr vcu_t MAX_VCUS = 1000;
constexpr mem_t MAX_MEM = std::numeric_limits<mem_t>::max();
constexpr timecount_t MAX_TIME = std::numeric_limits<timecount_t>::max();
constexpr copy_t MAX_COPIES = 10;

consteval mem_t operator""_KB(unsigned long long val) {
  return static_cast<mem_t>(val * BYTES_IN_POWER);
}
consteval mem_t operator""_MB(unsigned long long val) {
  return static_cast<mem_t>(val * BYTES_IN_POWER * BYTES_IN_POWER);
}
consteval mem_t operator""_GB(unsigned long long val) {
  return static_cast<mem_t>(val * BYTES_IN_POWER * BYTES_IN_POWER * BYTES_IN_POWER);
}

consteval timecount_t operator""_us(unsigned long long val) {
  return static_cast<timecount_t>(val);
}
consteval timecount_t operator""_ms(unsigned long long val) {
  return static_cast<timecount_t>(val * 1000);
}
consteval timecount_t operator""_s(unsigned long long val) {
  return static_cast<timecount_t>(val * 1000 * 1000);
}

consteval vcu_t operator""_vcus(long double val) {
  // Return fraction of MAX_VCUS
  return static_cast<vcu_t>(val * MAX_VCUS);
}

struct Resources {
  vcu_t vcu{};
  mem_t mem{};

  Resources() = default;
  Resources(vcu_t vcu, mem_t mem) : vcu(vcu), mem(mem) {
  }

  [[nodiscard]] bool empty() const {
    return vcu == 0 && mem == 0;
  }

  [[nodiscard]] bool empty_vcu() const {
    return vcu == 0;
  }

  [[nodiscard]] bool empty_mem() const {
    return mem == 0;
  }

  Resources &operator+=(const Resources &rhs) {
    vcu += rhs.vcu;
    mem += rhs.mem;
    return *this;
  }

  Resources &operator-=(const Resources &rhs) {
    vcu -= rhs.vcu;
    mem -= rhs.mem;
    return *this;
  }

  Resources operator+(const Resources &rhs) const {
    Resources result = *this;
    result += rhs;
    return result;
  }

  Resources operator-(const Resources &rhs) const {
    Resources result = *this;
    result -= rhs;
    return result;
  }

  [[nodiscard]] bool operator==(const Resources &rhs) const {
    return vcu == rhs.vcu && mem == rhs.mem;
  }

  [[nodiscard]] bool operator!=(const Resources &rhs) const {
    return !(*this == rhs);
  }

  [[nodiscard]] bool operator<(const Resources &rhs) const {
    return vcu < rhs.vcu && mem < rhs.mem;
  }

  [[nodiscard]] bool operator>(const Resources &rhs) const {
    return vcu > rhs.vcu && mem > rhs.mem;
  }

  [[nodiscard]] bool operator<=(const Resources &rhs) const {
    return vcu <= rhs.vcu && mem <= rhs.mem;
  }

  [[nodiscard]] bool operator>=(const Resources &rhs) const {
    return vcu >= rhs.vcu && mem >= rhs.mem;
  }

  friend std::ostream &operator<<(std::ostream &os, const Resources &r) {
    os << "Resources(VCUs: " << r.vcu << ", MEM: " << r.mem << ")";
    return os;
  }
};
