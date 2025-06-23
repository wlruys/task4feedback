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


#nanobind_build_library(nanobind-static)
#add_library(fastsim2 MODULE ${bindings_src})

#target_link_libraries(fastsim2 PRIVATE nanobind-static)
nanobind_add_module(fastsim2 NB_STATIC ${bindings_src})

target_include_directories(fastsim2
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

set_target_properties(fastsim2 PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
)

target_link_libraries(fastsim2 PRIVATE backend python_deps project_warnings)
target_compile_features(fastsim2 PUBLIC cxx_std_20)

check_cxx_compiler_flag("-fno-modules-ts" HAS_FNO_MODULES_TS)
if (HAS_FNO_MODULES_TS)
    target_compile_options(fastsim2 PRIVATE -fno-modules-ts)
endif()

if(CMAKE_CROSSCOMPILING AND SKBUILD_CROSS_STATIC_LIBCPP)
    nanobind_musl_static_libcpp(fastsim2)
endif()

if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
    nanobind_lto(fastsim2)
    nanobind_strip(fastsim2)
endif()

# nanobind_add_stub(fastsim2_stub
#     MODULE fastsim2
#     OUTPUT fastsim2.pyi
#     MARKER_FILE py.typed
#     DEPENDS fastsim2)

nanobind_opt_size(fastsim2)
nanobind_lto(fastsim2)
nanobind_set_visibility(fastsim2)
nanobind_strip(fastsim2)
nanobind_extension(fastsim2)
nanobind_compile_options(fastsim2)
nanobind_link_options(fastsim2)
nanobind_musl_static_libcpp(fastsim2)


install(TARGETS fastsim2
        COMPONENT python
        LIBRARY DESTINATION task4feedback)

# install(FILES  ${CMAKE_CURRENT_BINARY_DIR}/fastsim2.pyi
#                ${CMAKE_CURRENT_BINARY_DIR}/py.typed
#         COMPONENT python
#         DESTINATION task4feedback)
