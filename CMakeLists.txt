cmake_minimum_required(VERSION 3.22)
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
option(ENABLE_MIMALLOC      "Use mimalloc memory allocator"          OFF)
option(ENABLE_NATIVE_TUNING "Add -march=native / -mtune=native"      OFF)
option(PGO_MODE "PGO mode: Generate / Use / None"       "None")
option(SIM_RECORD  "Enable resource-usage and data-location logging"  OFF)
option(ENABLE_CCACHE       "Enable ccache for faster builds" OFF)
option(ENABLE_TRACY         "Enable Tracy profiler support"   OFF)
option(ENABLE_TRACY_LIBDEBUG "Enable Tracy to fetch system debug in" OFF)
option(ENABLE_PARMETIS "Enable ParMETIS, everything needs to be installed in ~/local" OFF)
option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)


if(NOT DEFINED FETCHCONTENT_UPDATES_DISCONNECTED)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Disable FetchContent updates by default" FORCE)
endif()

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


# Make martinus/unordered_dense ankerl::unordered_map available
set(ANKERL_UNORDERED_DENSE_LOCAL_PATH "" CACHE PATH "Path to local unordered_dense (to avoid network fetch)")
if(NOT ANKERL_UNORDERED_DENSE_LOCAL_PATH AND EXISTS "${CMAKE_SOURCE_DIR}/external/unordered_dense")
    set(ANKERL_UNORDERED_DENSE_LOCAL_PATH "${CMAKE_SOURCE_DIR}/external/unordered_dense")
endif()

if(ANKERL_UNORDERED_DENSE_LOCAL_PATH)
    message(STATUS "Using local unordered_dense at ${ANKERL_UNORDERED_DENSE_LOCAL_PATH}")
    FetchContent_Declare(
        ankerl_unordered_dense
        SOURCE_DIR ${ANKERL_UNORDERED_DENSE_LOCAL_PATH}
    )
else()
    FetchContent_Declare(
        ankerl_unordered_dense
        GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
        GIT_TAG        v4.5.0
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
    )
endif()
FetchContent_MakeAvailable(ankerl_unordered_dense)
if(TARGET unordered_dense)
    set_target_properties(unordered_dense PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
            "$<BUILD_INTERFACE:${ankerl_unordered_dense_SOURCE_DIR}/include>;$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
elseif(TARGET unordered_dense::unordered_dense)
    # Upstream defines only the namespaced alias; create a real target for export/install
    add_library(unordered_dense INTERFACE)
    target_link_libraries(unordered_dense INTERFACE unordered_dense::unordered_dense)
    set_target_properties(unordered_dense PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
            "$<BUILD_INTERFACE:${ankerl_unordered_dense_SOURCE_DIR}/include>;$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
endif()

if(TARGET unordered_dense)
    install(TARGETS unordered_dense EXPORT backendTargets)
endif()
install(DIRECTORY ${ankerl_unordered_dense_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


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


set(TRACY_LOCAL_PATH "" CACHE PATH "Path to a local Tracy checkout to avoid network fetches")
if(NOT TRACY_LOCAL_PATH AND EXISTS "${CMAKE_SOURCE_DIR}/external/tracy")
    set(TRACY_LOCAL_PATH "${CMAKE_SOURCE_DIR}/external/tracy")
endif()

add_library(tracy_headers INTERFACE)

if(ENABLE_TRACY)
    if(TRACY_LOCAL_PATH)
        message(STATUS "Using local Tracy source at ${TRACY_LOCAL_PATH}")
        FetchContent_Declare(tracy
            SOURCE_DIR ${TRACY_LOCAL_PATH})
    else()
        set(TRACY_VERSION_TAG v0.11.1)
        FetchContent_Declare(tracy
            GIT_REPOSITORY https://github.com/wolfpld/tracy.git
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
            GIT_TAG        ${TRACY_VERSION_TAG})
    endif()
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
    # Provide stub headers to avoid network dependency when Tracy is disabled
    target_compile_definitions(tracy_headers INTERFACE TRACY_ENABLE=0)
    target_include_directories(tracy_headers
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/tracy_stub>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
endif()

if(ENABLE_TRACY)
    target_include_directories(tracy_headers
        INTERFACE
            $<BUILD_INTERFACE:${tracy_SOURCE_DIR}/public>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tracy>)  # contains Tracy.hpp
    install(DIRECTORY ${tracy_SOURCE_DIR}/public/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tracy)
endif()

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

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

include_directories(${Python_INCLUDE_DIRS})

add_library(project_warnings INTERFACE)

if(ENABLE_ASAN)
    target_compile_options(project_warnings INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options   (project_warnings INTERFACE -fsanitize=address)
endif()

if(ENABLE_NATIVE_TUNING)
    target_compile_options(project_warnings INTERFACE -march=native -mtune=native)
endif()

target_compile_options(project_warnings INTERFACE -Wall -Wextra -Wpedantic)
if(WARNINGS_AS_ERRORS)
    target_compile_options(project_warnings INTERFACE -Werror)
endif()

if(ENABLE_TRACY)
    target_compile_options(project_warnings INTERFACE -fno-omit-frame-pointer)
endif()

add_subdirectory(src/backend)      # builds simulator library
add_subdirectory(src/bindings)     # builds python module 

include(CPack)                     #for packaging
include(GNUInstallDirs)            #for install dirs
include(CMakePackageConfigHelpers)

set(TASK4FEEDBACK_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/external/tracy_stub/tracy
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS project_warnings tracy_headers EXPORT backendTargets)

install(EXPORT backendTargets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${TASK4FEEDBACK_INSTALL_CMAKEDIR}
    FILE ${PROJECT_NAME}Targets.cmake)

configure_package_config_file(
    cmake/task4feedbackConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${TASK4FEEDBACK_INSTALL_CMAKEDIR}
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${TASK4FEEDBACK_INSTALL_CMAKEDIR}
)
