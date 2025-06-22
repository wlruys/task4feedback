cmake_minimum_required(VERSION 3.28)
include(FetchContent)

project(${SKBUILD_PROJECT_NAME}
        VERSION 0.2.0
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS  OFF)

# ---- project options -------------------------------------------------------
option(ENABLE_ASAN          "Compile with AddressSanitizer"          OFF)
option(ENABLE_NATIVE_TUNING "Add -march=native / -mtune=native"      OFF)
option(PGO_MODE "PGO mode: Generate / Use / None"       "None")
option(SIM_TRACK_RESOURCES  "Enable resource-usage logging"  ON)
option(SIM_TRACK_LOCATION   "Enable data-location logging"   ON)

add_library(tabulate INTERFACE)
target_include_directories(tabulate
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/tabulate/include)

find_package(spdlog CONFIG QUIET)

if(NOT spdlog_FOUND)
  include(FetchContent)

  set(SPDLOG_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
  set(SPDLOG_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
  set(SPDLOG_BUILD_BENCH     OFF CACHE BOOL "" FORCE)

  set(SPDLOG_BUILD_STATIC    ON  CACHE BOOL "" FORCE)
  set(SPDLOG_BUILD_SHARED    OFF CACHE BOOL "" FORCE)
  set(SPDLOG_BUILD_PIC       ON  CACHE BOOL "" FORCE)

  set(SPDLOG_FMT_EXTERNAL    ON  CACHE BOOL "" FORCE)

  FetchContent_Declare(
      spdlog
      GIT_REPOSITORY https://github.com/gabime/spdlog.git
      GIT_TAG        v1.14.1            # pin to a release; avoid 'v1.x'
  )

  FetchContent_MakeAvailable(spdlog)
endif()


set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LD_PGO_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_PGO_FLAGS}")


find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
add_library(python_deps INTERFACE)
target_link_libraries(python_deps
    INTERFACE Python::Python Python::NumPy)

find_package(nanobind CONFIG REQUIRED)

add_library(project_warnings INTERFACE)

if(ENABLE_ASAN)
    target_compile_options(project_warnings INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options   (project_warnings INTERFACE -fsanitize=address)
endif()

if(ENABLE_NATIVE_TUNING)
    target_compile_options(project_warnings INTERFACE -march=native -mtune=native)
endif()


add_subdirectory(src/backend)      # defines backend target
add_subdirectory(src/bindings)     # defines python module 

include(CPack)                     #for packaging
include(GNUInstallDirs)            #for install dirs
