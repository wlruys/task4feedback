#pragma once

template <typename T> constexpr int to_int(T value) {
  return static_cast<int>(value);
}