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

find_library(GKLIB_LIBRARY
  NAMES GKlib
  PATHS $ENV{HOME}/local/lib
  NO_DEFAULT_PATH
)
if(NOT GKLIB_LIBRARY)
  message(FATAL_ERROR "GKlib library not found in $ENV{HOME}/local/lib")
endif()

add_library(GKlib::GKlib SHARED IMPORTED)
set_target_properties(GKlib::GKlib PROPERTIES
  IMPORTED_LOCATION            "${GKLIB_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HOME}/local/include"
)

find_library(METIS_LIBRARY
  NAMES metis METIS Metis
  PATHS $ENV{HOME}/local/lib
  NO_DEFAULT_PATH
)
if(NOT METIS_LIBRARY)
  message(FATAL_ERROR "METIS library not found in $ENV{HOME}/local/lib")
endif()

add_library(METIS::METIS SHARED IMPORTED)
set_target_properties(METIS::METIS PROPERTIES
  IMPORTED_LOCATION            "${METIS_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HOME}/local/include"
)

find_library(PARMETIS_LIBRARY
  NAMES parmetis
  PATHS $ENV{HOME}/local/lib
  NO_DEFAULT_PATH
)
if(NOT PARMETIS_LIBRARY)
  message(FATAL_ERROR "ParMETIS library not found in $ENV{HOME}/local/lib")
endif()

add_library(ParMETIS::ParMETIS STATIC IMPORTED)
set_target_properties(ParMETIS::ParMETIS PROPERTIES
  IMPORTED_LOCATION "${PARMETIS_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HOME}/local/include"
)

set_target_properties(fastsim2 PROPERTIES
  INSTALL_RPATH "$ENV{HOME}/local/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)
endif()

if(ENABLE_PARMETIS)
  # make sure you’ve already done find_package(MPI ...), find_package(ParMETIS ...), etc.
  target_link_libraries(fastsim2
    PRIVATE
      MPI::MPI_CXX
      ParMETIS::ParMETIS
      METIS::METIS
      GKlib::GKlib
  )
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
