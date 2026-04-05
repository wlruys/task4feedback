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
        include/kahypar_wrapper.hpp
        include/iterator.hpp
        include/macros.hpp
        include/metis_wrapper.hpp
        include/noise.hpp
        include/queues.hpp
        include/resources.hpp
        include/eviction_policy.hpp
        include/mappers.hpp
        include/scheduler.hpp
        include/scheduler_state.hpp
        include/transition_conditions.hpp
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

target_compile_definitions(backend
  PUBLIC
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:RelWithDebInfo>:DEBUG>
)

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

if(ENABLE_PARMETIS)
  target_compile_definitions(backend PUBLIC ENABLE_PARMETIS)
endif()

if(ENABLE_METIS)
  set(METIS_PREFIX $ENV{METIS_INSTALL_PATH})
  if(NOT METIS_PREFIX)
    message(FATAL_ERROR "ENV METIS_INSTALL_PATH is not set. Set it to the install prefix that contains include/ and lib/.")
  endif()

  find_path(METIS_INCLUDE_DIR NAMES metis.h HINTS ${METIS_PREFIX}/include)
  find_library(METIS_LIBRARY NAMES metis METIS HINTS ${METIS_PREFIX}/lib)

  if(NOT METIS_INCLUDE_DIR OR NOT METIS_LIBRARY)
    message(FATAL_ERROR "Could not find METIS under ${METIS_PREFIX}.\n"
                        "Checked ${METIS_PREFIX}/include and ${METIS_PREFIX}/lib.")
  endif()

  if(NOT TARGET METIS::METIS)
    add_library(METIS::METIS SHARED IMPORTED)
    set_target_properties(METIS::METIS PROPERTIES
      IMPORTED_LOCATION "${METIS_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIR}")
  endif()

  target_link_libraries(backend PUBLIC METIS::METIS)
  target_include_directories(backend PUBLIC ${METIS_INCLUDE_DIR})
  target_compile_definitions(backend PUBLIC ENABLE_METIS)
endif()

if(ENABLE_KAHYPAR)
  set(KAHYPAR_PREFIX $ENV{KAHYPAR_INSTALL_PATH})
  if(NOT KAHYPAR_PREFIX)
    message(FATAL_ERROR "ENV KAHYPAR_INSTALL_PATH is not set. Set it to the install prefix that contains include/ and lib/.")
  endif()

  set(KAHYPAR_CONFIG_FILE "$ENV{KAHYPAR_CONFIG_PATH}")
  if(NOT KAHYPAR_CONFIG_FILE)
    set(KAHYPAR_CONFIG_FILE "${KAHYPAR_PREFIX}/config/km1_kKaHyPar_sea20.ini")
  endif()

  find_path(KAHYPAR_INCLUDE_DIR NAMES libkahypar.h HINTS ${KAHYPAR_PREFIX}/include)
  find_library(KAHYPAR_LIBRARY NAMES kahypar libkahypar HINTS ${KAHYPAR_PREFIX}/lib)

  if(NOT KAHYPAR_INCLUDE_DIR OR NOT KAHYPAR_LIBRARY)
    message(FATAL_ERROR "Could not find KaHyPar under ${KAHYPAR_PREFIX}.\n"
                        "Checked ${KAHYPAR_PREFIX}/include and ${KAHYPAR_PREFIX}/lib.")
  endif()

  if(NOT EXISTS "${KAHYPAR_CONFIG_FILE}")
    message(FATAL_ERROR "Could not find KaHyPar config file at ${KAHYPAR_CONFIG_FILE}.\n"
                        "Set ENV KAHYPAR_CONFIG_PATH to a valid .ini file such as km1_kKaHyPar_sea20.ini.")
  endif()

  if(NOT TARGET KaHyPar::kahypar)
    add_library(KaHyPar::kahypar SHARED IMPORTED)
    set_target_properties(KaHyPar::kahypar PROPERTIES
      IMPORTED_LOCATION "${KAHYPAR_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${KAHYPAR_INCLUDE_DIR}")
  endif()

  target_link_libraries(backend PUBLIC KaHyPar::kahypar)
  target_compile_definitions(backend PUBLIC ENABLE_KAHYPAR)
  target_compile_definitions(backend PUBLIC KAHYPAR_CONFIG_PATH=\"${KAHYPAR_CONFIG_FILE}\")
endif()

target_include_directories(backend
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(TARGETS backend)
