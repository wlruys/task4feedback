cmake_minimum_required(VERSION 3.29)
include(FetchContent)
include(CheckCXXCompilerFlag)

project(${SKBUILD_PROJECT_NAME}
        VERSION 0.2.0
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS  OFF)
set(CMAKE_UNITY_BUILD_BATCH_SIZE 0)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- project options -------------------------------------------------------
option(ENABLE_ASAN          "Compile with AddressSanitizer"          OFF)
option(ENABLE_MIMALLOC      "Use mimalloc memory allocator"          ON)
option(ENABLE_NATIVE_TUNING "Add -march=native / -mtune=native"      ON)
option(PGO_MODE "PGO mode: Generate / Use / None"       "None")
option(SIM_TRACK_RESOURCES  "Enable resource-usage logging"  ON)
option(SIM_TRACK_LOCATION   "Enable data-location logging"   ON)
option(ENABLE_CCACHE       "Enable ccache for faster builds" OFF)
option(ENABLE_TRACY         "Enable Tracy profiler support"   OFF)
option(ENABLE_TRACY_LIBDEBUG "Enable Tracy to fetch system debug in" OFF)


if(ENABLE_CCACHE)
    find_program(CCACHE_FOUND ccache)
    if(CCACHE_FOUND)
        message(STATUS "Using ccache for faster builds")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK   ccache)
    else()
        message(WARNING "ccache not found, disabling ENABLE_CCACHE")
        set(ENABLE_CCACHE OFF)
    endif()
endif()


# Make martinus/unordered_dense ankerl::unordereed_map available
FetchContent_Declare(
    ankerl_unordered_dense
    GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
    GIT_TAG        v4.5.0
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
)
FetchContent_MakeAvailable(ankerl_unordered_dense)






if(ENABLE_MIMALLOC AND NOT ENABLE_ASAN)  # Don't use with AddressSanitizer
    find_package(mimalloc CONFIG QUIET)
    if(mimalloc_FOUND)
        message(STATUS "Using system-installed mimalloc")
        add_library(mimalloc::mimalloc ALIAS mimalloc)
    else()
        message(STATUS "Fetching mimalloc from GitHub")
        FetchContent_Declare(
            mimalloc
            GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
            GIT_TAG        v2.1.7
        )
        
        # Configure mimalloc build options
        set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
        set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
        set(MI_BUILD_STATIC ON CACHE BOOL "" FORCE)
        set(MI_OVERRIDE ON CACHE BOOL "" FORCE)
        

        FetchContent_MakeAvailable(mimalloc)
        add_library(mimalloc::mimalloc ALIAS mimalloc-static)
    endif()
    
    message(STATUS "mimalloc enabled - automatic malloc/free override")
endif()


FetchContent_Declare(tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_SHALLOW    TRUE
    GIT_PROGRESS TRUE
    GIT_TAG master
)     

add_library(tracy_headers INTERFACE)


if(ENABLE_TRACY)
    set(TRACY_ENABLE ON CACHE BOOL "" FORCE)
    set(TRACY_ON_DEMAND ON CACHE BOOL "" FORCE)
    set(TRACY_LTO ON CACHE BOOL "" FORCE)
    set(TRACY_NO_SAMPLING ON CACHE BOOL "" FORCE)
    if(ENABLE_TRACY_LIBDEBUG)
        set(TRACY_DEBUGINFOD ON CACHE BOOL "" FORCE)
    else()
        set(TRACY_DEBUGINFOD OFF CACHE BOOL "" FORCE)
    endif()
    FetchContent_MakeAvailable(tracy)     

    target_link_libraries(tracy_headers    
        INTERFACE Tracy::TracyClient)
    target_compile_definitions(tracy_headers
        INTERFACE TRACY_ENABLE TRACY_LTO TRACY_ON_DEMAND)
    target_compile_options(tracy_headers
        INTERFACE -fno-omit-frame-pointer -g)
    if(ENABLE_TRACY_LIBDEBUG)
        target_compile_definitions(tracy_headers 
            INTERFACE TRACY_DEBUGINFOD)
        include(FindPkgConfig)
        pkg_check_modules(debuginfod REQUIRED libdebuginfod)
        target_include_directories(tracy_headers INTERFACE ${debuginfod_INCLUDE_DIRS})
        target_link_libraries(tracy_headers INTERFACE ${debuginfod_LINK_LIBRARIES})   
    endif()
else()
    set(TRACY_ENABLE OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(tracy)     
endif()

target_include_directories(tracy_headers
    INTERFACE ${tracy_SOURCE_DIR}/public)  # contains Tracy.hpp

find_package(fmt CONFIG QUIET)
if(fmt_FOUND)
    message(STATUS "Using system-installed fmt ${fmt_VERSION}")
else()
    FetchContent_Declare(
      fmt
      GIT_REPOSITORY https://github.com/fmtlib/fmt.git
      GIT_TAG        10.2.1
      GIT_SHALLOW    TRUE
      GIT_PROGRESS   TRUE
    )
    # build fmt as static with PIC by default
    set(FMT_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(FMT_TEST OFF     CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(fmt)
endif()


find_package(spdlog CONFIG QUIET)
if(spdlog_FOUND)
  message(STATUS "Using system-installed spdlog ${spdlog_VERSION}")
else()

  set(SPDLOG_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
  set(SPDLOG_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
  set(SPDLOG_BUILD_BENCH     OFF CACHE BOOL "" FORCE)
  #
  # set(SPDLOG_BUILD_STATIC    ON  CACHE BOOL "" FORCE)
  # set(SPDLOG_BUILD_SHARED    OFF CACHE BOOL "" FORCE)
  # set(SPDLOG_BUILD_PIC       ON  CACHE BOOL "" FORCE)
    set(FMT_SYSTEM_HEADERS ON CACHE BOOL "" FORCE)
  set(SPDLOG_HEADER_ONLY ON CACHE BOOL "" FORCE)
  set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)

  FetchContent_Declare(
      spdlog
      GIT_REPOSITORY https://github.com/gabime/spdlog.git
      GIT_TAG        v1.14.1
      GIT_SHALLOW    TRUE
      GIT_PROGRESS   TRUE          
  )

  FetchContent_MakeAvailable(spdlog)
endif()


if(PGO_MODE STREQUAL "Generate")
    add_compile_options(-fprofile-generate)
    add_link_options   (-fprofile-generate)
elseif(PGO_MODE STREQUAL "Use")
    add_compile_options(-fprofile-use -fprofile-correction)
    add_link_options   (-fprofile-use)
endif()

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

include_directories(${Python_INCLUDE_DIRS})
include_directories(${Python_NumPy_INCLUDE_DIRS})

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
