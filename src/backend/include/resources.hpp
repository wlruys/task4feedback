#pragma once
#include "devices.hpp"
#include "settings.hpp"
#include <array>

using vcu_t = unsigned int;
using mem_t = unsigned long long;
using timecount_t = unsigned long long;
using copy_t = unsigned int;

class Variant {
public:
  DeviceType arch = DeviceType::NONE;
  vcu_t vcu = 0;
  mem_t mem = 0;
  timecount_t time = 0;

  Variant() = default;
  Variant(DeviceType arch, vcu_t vcu, mem_t mem, timecount_t time)
      : arch(arch), vcu(vcu), mem(mem), time(time) {}
};

using VariantList = std::array<Variant, num_device_types>;