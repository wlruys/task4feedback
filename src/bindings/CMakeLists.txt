find_package(nanobind CONFIG REQUIRED)

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  fastsim2
  STABLE_ABI
  NB_STATIC
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


nanobind_add_stub(
  fastsim2_stub
  MODULE fastsim2
  OUTPUT fastsim2.pyi 
  PYTHON_PATH $<TARGET_FILE_DIR:fastsim2>
  MARKER_FILE py.typed
  DEPENDS fastsim2
  VERBOSE
)

target_link_libraries(fastsim2 PRIVATE backend)
target_include_directories(fastsim2 PRIVATE ${PROJECT_SOURCE_DIR}/src/backend)
target_include_directories(fastsim2 PRIVATE ${PROJECT_SOURCE_DIR}/src/backend/include)
target_include_directories(fastsim2 PRIVATE ${NumPy_INCLUDE_DIRS})
install(TARGETS fastsim2 LIBRARY DESTINATION task4feedback)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fastsim2.pyi DESTINATION task4feedback)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/py.typed DESTINATION task4feedback)

