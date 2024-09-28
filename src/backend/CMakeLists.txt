set(BACKEND_SRC 
    graph.cpp
    tasks.cpp
    task_manager.cpp 
    queues.cpp 
)

set(BACKEND_HDR
    data.hpp
    devices.hpp
    graph.hpp
    macros.hpp 
    queues.hpp 
    resources.hpp
    scheduler.hpp 
    settings.hpp 
    tasks.hpp 
    task_manager.hpp
    device_manager.hpp
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

#${PYTHON_RELATIVE_SITE_PACKAGES_DIR}/parla/lib/

install(TARGETS backend
    EXPORT SimulatorBackendTargets
    LIBRARY DESTINATION .
)
