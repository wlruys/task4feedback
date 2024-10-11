
enable_testing()

include(${3RD_PARTY_DIR}/doctest/doctest.cmake)


list(APPEND SIMULATOR_UNITTESTS
  test_queue
  test_tasks
  test_noise
)

foreach(unittest IN LISTS SIMULATOR_UNITTESTS)
  add_executable(${unittest} ${unittest}.cpp)
  target_link_libraries(${unittest} backend)
  target_compile_options(${unittest} PRIVATE -O3)
  target_include_directories(${unittest} PRIVATE ${3RD_PARTY_DIR}/doctest)
  target_include_directories(${unittest} PRIVATE ${PROJECT_SOURCE_DIR}/src/backend)
  target_include_directories(${unittest} PRIVATE ${PROJECT_SOURCE_DIR}/src/backend/include)
  doctest_discover_tests(${unittest})
  install(TARGETS ${unittest} DESTINATION ${CMAKE_INSTALL_PREFIX}/testing/c/)
endforeach()
