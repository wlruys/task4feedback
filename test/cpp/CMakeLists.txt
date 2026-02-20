add_executable(backend_eviction_runtime_tests
    test_eviction_runtime_refactor.cpp
)

target_link_libraries(backend_eviction_runtime_tests
    PRIVATE backend
)

target_include_directories(backend_eviction_runtime_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/external/doctest
)

add_test(NAME backend_eviction_runtime_tests COMMAND backend_eviction_runtime_tests)
