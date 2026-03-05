#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "include/macros.hpp"

TEST_CASE("T4F_ENABLE_DEBUG_CHECKS matches DEBUG definition") {
#if defined(DEBUG)
  CHECK(T4F_ENABLE_DEBUG_CHECKS == 1);
#else
  CHECK(T4F_ENABLE_DEBUG_CHECKS == 0);
#endif
}

TEST_CASE("T4F_DEBUG_CHECK expression evaluation respects build mode") {
  int side_effect_counter = 0;
  T4F_DEBUG_CHECK(++side_effect_counter == 1);

#if T4F_ENABLE_DEBUG_CHECKS
  CHECK(side_effect_counter == 1);
#else
  CHECK(side_effect_counter == 0);
#endif
}

TEST_CASE("T4F_DEBUG_ONLY compiles out in non-debug-check builds") {
  int side_effect_counter = 0;
  T4F_DEBUG_ONLY(++side_effect_counter;);

#if T4F_ENABLE_DEBUG_CHECKS
  CHECK(side_effect_counter == 1);
#else
  CHECK(side_effect_counter == 0);
#endif
}
