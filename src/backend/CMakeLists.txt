add_library(backend STATIC)

target_sources(backend
  PRIVATE
    queues.cpp
    scheduler.cpp
    tasks.cpp
)

target_sources(backend
  PUBLIC
    FILE_SET cxx_hdr TYPE HEADERS
      BASE_DIRS include
      FILES
        include/action.hpp
        include/breakpoints.hpp
        include/data.hpp
        include/devices.hpp
        include/communication.hpp
        include/events.hpp
        include/iterator.hpp
        include/macros.hpp
        include/noise.hpp
        include/queues.hpp
        include/resources.hpp
        include/scheduler.hpp
        include/settings.hpp
        include/simulator.hpp
        include/stats.hpp
        include/tasks.hpp
        include/parmetis_wrapper.hpp
)

target_compile_features(backend PUBLIC cxx_std_20)
set_target_properties(backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  PUBLIC_HEADER_SET cxx_hdr
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
  UNITY_BUILD ON
)

target_precompile_headers(backend PRIVATE
    <vector>
    <string>
    <memory>
    <algorithm>
    <unordered_map>
    <spdlog/spdlog.h>
    <fmt/format.h>
)

target_compile_definitions(backend
  PUBLIC
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:RelWithDebInfo>:DEBUG>
    T4F_MAX_DEVICES=${T4F_MAX_DEVICES}
)

target_link_libraries(backend
  PUBLIC
    project_warnings
    spdlog::spdlog_header_only
    fmt::fmt
    tracy_headers
    unordered_dense::unordered_dense
)

if(SIM_RECORD)
    target_compile_definitions(backend PUBLIC SIM_RECORD)
endif()

if(DEFINED SPDLOG_ACTIVE_LEVEL)
    target_compile_definitions(backend PUBLIC SPDLOG_ACTIVE_LEVEL=${SPDLOG_ACTIVE_LEVEL})
else()
    target_compile_definitions(backend PUBLIC $<$<CONFIG:Release>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO>)
endif()

if(ENABLE_VECTORIZATION_REPORT)
    target_compile_options(backend PRIVATE
        $<$<CXX_COMPILER_ID:Clang>:-Rpass=loop-vectorize -Rpass-missed=loop-vectorize>
        $<$<CXX_COMPILER_ID:GNU>:-fopt-info-vec-optimized -fopt-info-vec-missed>
    )
endif()

if(ENABLE_MIMALLOC AND NOT ENABLE_ASAN)
    target_link_libraries(backend PRIVATE mimalloc::mimalloc)
    target_compile_definitions(backend PRIVATE MI_MALLOC_OVERRIDE=1)
endif()

if(ENABLE_PARMETIS)
  target_compile_definitions(backend PUBLIC ENABLE_PARMETIS)
endif()

target_include_directories(backend
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(TARGETS backend)
