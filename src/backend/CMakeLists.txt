add_library(backend STATIC)
target_sources(backend
  PRIVATE
    data_manager.cpp
    device_manager.cpp
    graph.cpp
    queues.cpp
    scheduler.cpp
    task_manager.cpp
    tasks.cpp
)

target_sources(backend
  PUBLIC
    FILE_SET cxx_hdr TYPE HEADERS
      BASE_DIRS include
      FILES
        include/action.hpp
        include/breakpoints.hpp
        include/data_manager.hpp
        include/device_manager.hpp
        include/communication_manager.hpp
        include/observer.hpp
        include/devices.hpp
        include/event_manager.hpp
        include/events.hpp
        include/graph.hpp
        include/iterator.hpp
        include/macros.hpp
        include/noise.hpp
        include/queues.hpp
        include/resources.hpp
        include/scheduler.hpp
        include/settings.hpp
        include/simulator.hpp
        include/stats.hpp
        include/task_manager.hpp
        include/tasks.hpp
        include/features.hpp
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


find_package(Python COMPONENTS Development NumPy REQUIRED)
target_link_libraries(backend PUBLIC Python::Python Python::NumPy tabulate python_deps project_warnings spdlog::spdlog)


if(SIM_TRACK_RESOURCES)
    target_compile_definitions(backend PUBLIC SIM_TRACK_RESOURCES)
endif()

if(SIM_TRACK_LOCATION)
    target_compile_definitions(backend PUBLIC SIM_TRACK_LOCATION)
endif()

if(DEFINED SPDLOG_ACTIVE_LEVEL)
    target_compile_definitions(backend PUBLIC SPDLOG_ACTIVE_LEVEL=${SPDLOG_ACTIVE_LEVEL})
endif()


target_include_directories(backend
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(TARGETS backend)