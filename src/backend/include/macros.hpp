#pragma once
#include <cstdio>
#include <cstdlib>

template <typename T> constexpr int to_int(T value) {
  return static_cast<int>(value);
}

#define MON_Internal_UnusedStringify(macro_arg_string_literal) #macro_arg_string_literal

#define MONUnusedParameter(macro_arg_parameter)                                                    \
  _Pragma(MON_Internal_UnusedStringify(unused(macro_arg_parameter)))

#if defined(DEBUG)
#define T4F_ENABLE_DEBUG_CHECKS 1
#else
#define T4F_ENABLE_DEBUG_CHECKS 0
#endif

namespace t4f::detail {
[[noreturn]] inline void debug_check_fail(const char *expr, const char *file, int line) {
  std::fprintf(stderr, "T4F debug check failed: (%s) at %s:%d\n", expr, file, line);
  std::abort();
}
} // namespace t4f::detail

#if T4F_ENABLE_DEBUG_CHECKS
#define T4F_DEBUG_CHECK(expr)                                                                      \
  do {                                                                                             \
    if (!(expr)) {                                                                                 \
      ::t4f::detail::debug_check_fail(#expr, __FILE__, __LINE__);                                 \
    }                                                                                              \
  } while (0)

#define T4F_DEBUG_ONLY(...)                                                                        \
  do {                                                                                             \
    __VA_ARGS__;                                                                                   \
  } while (0)
#else
#define T4F_DEBUG_CHECK(expr) ((void)0)
#define T4F_DEBUG_ONLY(...) ((void)0)
#endif

#define T4F_PRECONDITION(expr) T4F_DEBUG_CHECK(expr)
#define T4F_POSTCONDITION(expr) T4F_DEBUG_CHECK(expr)
#define T4F_INVARIANT(expr) T4F_DEBUG_CHECK(expr)
#define T4F_ASSERT(expr) T4F_DEBUG_CHECK(expr)
