
add_library(fastsim2 MODULE 
  bindings.cpp
  event_bindings.cpp
  device_bindings.cpp
  task_bindings.cpp
  graph_bindings.cpp 
  data_bindings.cpp
  task_noise_bindings.cpp
  topology_bindings.cpp
  mapper_bindings.cpp
  scheduler_bindings.cpp
  simulator_bindings.cpp
  observer_bindings.cpp
)

target_link_libraries(fastsim2 PRIVATE backend)
target_link_libraries(fastsim2 PRIVATE nanobind-static)
target_include_directories(fastsim2 PRIVATE ${PROJECT_SOURCE_DIR}/src/backend)
target_include_directories(fastsim2 PRIVATE ${PROJECT_SOURCE_DIR}/src/backend/include)

nanobind_opt_size(fastsim2)
nanobind_lto(fastsim2)
nanobind_set_visibility(fastsim2)
nanobind_strip(fastsim2)
nanobind_extension(fastsim2)
nanobind_compile_options(fastsim2)
nanobind_link_options(fastsim2)
nanobind_musl_static_libcpp(fastsim2)

# nanobind_add_module(
#   # Name of the extension
#   fastsim2
#   STABLE_ABI
#   NB_STATIC
#   bindings.cpp
#   event_bindings.cpp
#   device_bindings.cpp
#   task_bindings.cpp
#   graph_bindings.cpp 
#   data_bindings.cpp
#   task_noise_bindings.cpp
#   topology_bindings.cpp
#   mapper_bindings.cpp
#   scheduler_bindings.cpp
#   simulator_bindings.cpp
#   observer_bindings.cpp
# )


nanobind_add_stub(
  fastsim2_stub
  MODULE fastsim2
  OUTPUT fastsim2.pyi 
  PYTHON_PATH $<TARGET_FILE_DIR:fastsim2>
  MARKER_FILE py.typed
  DEPENDS fastsim2
  VERBOSE
)

install(TARGETS fastsim2 LIBRARY DESTINATION task4feedback)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fastsim2.pyi DESTINATION task4feedback)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/py.typed DESTINATION task4feedback)

