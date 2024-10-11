#pragma once

template <typename T> constexpr int to_int(T value) {
  return static_cast<int>(value);
}

#define MON_Internal_UnusedStringify(macro_arg_string_literal)                 \
#macro_arg_string_literal

#define MONUnusedParameter(macro_arg_parameter)                                \
  _Pragma(MON_Internal_UnusedStringify(unused(macro_arg_parameter)))