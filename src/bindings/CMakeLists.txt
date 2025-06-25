find_package(nanobind REQUIRED CONFIG)


set(bindings_src
    bindings.cpp
    event_bindings.cpp
    device_bindings.cpp
    task_bindings.cpp
    graph_bindings.cpp
    data_bindings.cpp
    task_noise_bindings.cpp
    topology_bindings.cpp
    mapper_bindings.cpp
    scheduler_bindings.cpp
    simulator_bindings.cpp
    observer_bindings.cpp)

nanobind_add_module(fastsim2 NB_STATIC ${bindings_src})

target_include_directories(fastsim2
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

set_target_properties(fastsim2 PROPERTIES
  UNITY_BUILD ON
)

set_target_properties(fastsim2 PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
)

target_link_libraries(fastsim2 PRIVATE backend project_warnings tracy_headers unordered_dense::unordered_dense)
target_compile_features(fastsim2 PUBLIC cxx_std_20)

check_cxx_compiler_flag("-fno-modules-ts" HAS_FNO_MODULES_TS)
if (HAS_FNO_MODULES_TS)
    target_compile_options(fastsim2 PRIVATE -fno-modules-ts)
endif()

if(CMAKE_CROSSCOMPILING AND SKBUILD_CROSS_STATIC_LIBCPP)
    nanobind_musl_static_libcpp(fastsim2)
endif()

if(ENABLE_MIMALLOC AND NOT ENABLE_ASAN)
    target_link_libraries(fastsim2 PRIVATE mimalloc::mimalloc)
    target_compile_definitions(fastsim2 PRIVATE MI_MALLOC_OVERRIDE=1)
endif()

# if(CMAKE_BUILD_TYPE STREQUAL "Release")
#     nanobind_lto(fastsim2)
#     nanobind_strip(fastsim2)
# endif()

# Remove -Os -DNDEBUG from the fastsim2 target
# This is necessary to avoid issues with nanobind and the -Os optimization level.

target_link_options(fastsim2 PRIVATE -Wl,--no-gc-sections)
target_compile_options(fastsim2 PRIVATE -fno-function-sections -fno-data-sections)
set_target_properties(fastsim2 PROPERTIES
                      CXX_VISIBILITY_PRESET default
                      VISIBILITY_INLINES_HIDDEN OFF)


# nanobind_add_stub(fastsim2_stub
#     MODULE fastsim2
#     OUTPUT fastsim2.pyi
#     MARKER_FILE py.typed
#     DEPENDS fastsim2)

install(TARGETS fastsim2
        COMPONENT python
        LIBRARY DESTINATION task4feedback)
