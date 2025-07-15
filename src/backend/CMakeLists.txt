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
)

                                         
target_compile_features(backend PUBLIC cxx_std_20)
set_target_properties(backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ON 
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    PUBLIC_HEADER_SET          cxx_hdr
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

set_target_properties(backend PROPERTIES
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

if(CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_definitions(backend PUBLIC DEBUG)
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  target_compile_definitions(backend PUBLIC DEBUG)
endif()

target_link_libraries(backend PUBLIC project_warnings spdlog::spdlog_header_only fmt::fmt tracy_headers unordered_dense::unordered_dense)

if(SIM_RECORD)
    target_compile_definitions(backend PUBLIC SIM_RECORD)
endif()

if(DEFINED SPDLOG_ACTIVE_LEVEL)
    target_compile_definitions(backend PUBLIC SPDLOG_ACTIVE_LEVEL=${SPDLOG_ACTIVE_LEVEL})
endif()

if(ENABLE_MIMALLOC AND NOT ENABLE_ASAN)
    target_link_libraries(backend PRIVATE mimalloc::mimalloc)
    target_compile_definitions(backend PRIVATE MI_MALLOC_OVERRIDE=1)
endif()

if(ENABLE_TRACY)
    target_link_libraries(backend PUBLIC TracyClient)
    target_compile_definitions(backend PUBLIC ENABLE_TRACY TRACY_ENABLE TRACY_ON_DEMAND)
endif()

target_include_directories(backend
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(TARGETS backend)
