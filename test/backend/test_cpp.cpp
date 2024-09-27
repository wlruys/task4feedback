#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include <doctest.h>
#include <list>
#include <string>
#include <tuple>

#define DOCTEST_VALUE_PARAMETERIZED_DATA(case_idx, data, data_container)       \
  static size_t _doctest_subcase_idx = 0;                                      \
  std::for_each(data_container.begin(), data_container.end(),                  \
                [&](const auto &in) {                                          \
                  SUBCASE((std::string(#data_container "[") +                  \
                           std::to_string(_doctest_subcase_idx++) + "]")       \
                              .c_str()) {                                      \
                    data = in;                                                 \
                    case_idx = _doctest_subcase_idx;                           \
                  }                                                            \
                });                                                            \
  _doctest_subcase_idx = 0

TEST_CASE("Hello World") { CHECK(1 == 1); }