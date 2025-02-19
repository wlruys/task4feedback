set(BACKEND_SRC 
    data_manager.cpp 
    device_manager.cpp 
    graph.cpp 
    queues.cpp 
    scheduler.cpp 
    task_manager.cpp 
    tasks.cpp
)

set(BACKEND_HDR
    action.hpp
    breakpoints.hpp
    data_manager.hpp
    device_manager.hpp 
    communication_manager.hpp
    observer.hpp
    devices.hpp 
    event_manager.hpp
    events.hpp
    graph.hpp 
    iterator.hpp
    macros.hpp
    noise.hpp
    queues.hpp
    resources.hpp
    scheduler.hpp
    settings.hpp
    simulator.hpp
    stats.hpp
    task_manager.hpp
    tasks.hpp
)

# Add "include" to all header files
foreach(HEADER ${BACKEND_HDR})
    set(BACKEND_HDR_INCLUDE ${BACKEND_HDR_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/include/${HEADER})
endforeach()

message(STATUS "ALL FILES: ${BACKEND_SRC}")
add_library(backend STATIC ${BACKEND_SRC})

target_include_directories(backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(backend PROPERTIES PUBLIC_HEADER "${BACKEND_HDR_INCLUDE}")

message(STATUS "SIM_TRACK_LOCATION: ${SIM_TRACK_LOCATION}")
if(SIM_TRACK_LOCATION)
    message(STATUS "...data location logging enabled")
    target_compile_definitions(backend PUBLIC SIM_TRACK_LOCATION)
endif()

message(STATUS "SIM_TRACK_RESOURCES: ${SIM_TRACK_RESOURCES}")
if(SIM_TRACK_RESOURCES)
    message(STATUS "...resource logging enabled")
    target_compile_definitions(backend PUBLIC SIM_TRACK_RESOURCES)
endif()

target_compile_definitions(backend PUBLIC SPDLOG_ACTIVE_LEVEL=${SPDLOG_ACTIVE_LEVEL})

#${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/parla/lib/

install(TARGETS backend
    EXPORT SimulatorBackendTargets
    LIBRARY DESTINATION .
)
