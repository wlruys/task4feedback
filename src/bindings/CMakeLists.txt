find_package(nanobind REQUIRED CONFIG)


set(bindings_src
    bindings.cpp
    event_bindings.cpp
    device_bindings.cpp
    task_bindings.cpp
    data_bindings.cpp
    task_noise_bindings.cpp
    topology_bindings.cpp
    mapper_bindings.cpp
    scheduler_bindings.cpp
    feature_bindings.cpp
    simulator_bindings.cpp
    parmetis_wrapper_bindings.cpp)

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

if(ENABLE_PARMETIS)
  find_package(MPI REQUIRED)

  # Expect an install prefix with `include/` and `lib/`
  set(PARMETIS_PREFIX $ENV{PARMETIS_INSTALL_PATH})
  if(NOT PARMETIS_PREFIX)
    message(FATAL_ERROR "ENV PARMETIS_INSTALL_PATH is not set. Set it to the install prefix that contains include/ and lib/.")
  endif()

  # Locate headers explicitly under include/
  find_path(PARMETIS_INCLUDE_DIR NAMES parmetis.h HINTS ${PARMETIS_PREFIX}/include)
  find_path(METIS_INCLUDE_DIR    NAMES metis.h     HINTS ${PARMETIS_PREFIX}/include)

  # Locate libs explicitly under lib/
  find_library(PARMETIS_LIBRARY NAMES parmetis ParMETIS HINTS ${PARMETIS_PREFIX}/lib)
  find_library(METIS_LIBRARY    NAMES metis METIS       HINTS ${PARMETIS_PREFIX}/lib)
  find_library(GKLIB_LIBRARY    NAMES GKlib gklib       HINTS ${PARMETIS_PREFIX}/lib)

  if(NOT PARMETIS_INCLUDE_DIR OR NOT METIS_INCLUDE_DIR OR
     NOT PARMETIS_LIBRARY OR NOT METIS_LIBRARY OR NOT GKLIB_LIBRARY)
    message(FATAL_ERROR "Could not find ParMETIS/METIS/GKlib under ${PARMETIS_PREFIX}.\n"
                        "Checked ${PARMETIS_PREFIX}/include and ${PARMETIS_PREFIX}/lib.")
  endif()

  # Create imported targets so downstream tools (nanobind stubgen) see CMake targets
  if(NOT TARGET GKlib::GKlib)
    add_library(GKlib::GKlib SHARED IMPORTED)
    set_target_properties(GKlib::GKlib PROPERTIES
      IMPORTED_LOCATION "${GKLIB_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${PARMETIS_PREFIX}/include")
  endif()

  if(NOT TARGET METIS::METIS)
    add_library(METIS::METIS SHARED IMPORTED)
    set_target_properties(METIS::METIS PROPERTIES
      IMPORTED_LOCATION "${METIS_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${PARMETIS_PREFIX}/include")
  endif()

  if(NOT TARGET ParMETIS::ParMETIS)
    add_library(ParMETIS::ParMETIS SHARED IMPORTED)
    set_target_properties(ParMETIS::ParMETIS PROPERTIES
      IMPORTED_LOCATION "${PARMETIS_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${PARMETIS_PREFIX}/include")
  endif()

  target_include_directories(fastsim2 PRIVATE ${PARMETIS_INCLUDE_DIR} ${METIS_INCLUDE_DIR})
  target_link_libraries(fastsim2 PRIVATE MPI::MPI_CXX
                                        ParMETIS::ParMETIS
                                        METIS::METIS
                                        GKlib::GKlib)
endif()

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

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    nanobind_lto(fastsim2)
    nanobind_strip(fastsim2)
endif()


nanobind_add_stub(fastsim2_stub
    MODULE fastsim2
    OUTPUT fastsim2.pyi
    MARKER_FILE py.typed
    DEPENDS fastsim2)

install(TARGETS fastsim2
        COMPONENT python
        LIBRARY DESTINATION task4feedback)
