cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 20)

project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

set(3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/external)
option(SIMULATOR_DEBUG "Enables all debugging flags" ON)

set(CMAKE_VERBOSE_MAKEFILE ON)
include(CTest)

include_directories(${3RD_PARTY_DIR}/tabulate/include)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED
  )

set(CYTHON cython)
message(STATUS "Cython: " ${CYTHON})
#Test if cython is installed
if(NOT CYTHON)
  message(
    FATAL_ERROR
      "Cython is required to build Parla. Please install cython and try again."
    )
endif()
#Test if cython exists
execute_process(COMMAND ${CYTHON} --version ERROR_QUIET OUTPUT_VARIABLE CYTHON_VERSION)
if(NOT CYTHON_VERSION)
  message(
    FATAL_ERROR
      "Cython is required to build Parla. Please install cython and try again."
    )
endif()

message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "Python_NumPy_INCLUDE_DIRS: ${Python_NumPy_INCLUDE_DIRS}")
message(STATUS "Python: " ${Python_EXECUTABLE})
include_directories(${NUMPY_INCLUDE})
include_directories(${Python_INCLUDE_DIRS})
include_directories(${Python_NumPy_INCLUDE_DIRS})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CYTHON_FLAGS "--cplus;--fast-fail")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(SIMULATOR_DEBUG_C_COMPILE_FLAGS "-fno-omit-frame-pointer")
  set(SIMULATOR_DEBUG_CYTHON_FLAGS "--line-directives;-p;--gdb")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SIMULATOR_DEBUG_C_COMPILE_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SIMULATOR_DEBUG_C_COMPILE_FLAGS}")
  list(APPEND CYTHON_FLAGS ${SIMULATOR_DEBUG_CYTHON_FLAGS})
  list(APPEND CYTHON_FLAGS "-X;linetrace=True")
endif()

if(UNIX)
  if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else(NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  endif()
else(NOT UNIX)
  message(
    ERROR
    "Windows is not supported. Please raise an issue if you need Windows support."
  )
endif()

add_subdirectory(src/backend)
add_subdirectory(src/task4feedback/fastsim)

add_subdirectory(test)

set(test_path_file ${CMAKE_SOURCE_DIR}/test/run_tests.sh)
file(WRITE ${test_path_file} "export PARLA_TESTS=${CMAKE_BINARY_DIR}/test\n")
file(APPEND ${test_path_file} "ctest --test-dir $PARLA_TESTS\n")
