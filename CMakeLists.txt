cmake_minimum_required(VERSION 3.25)
project(KernelGoblin VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(KG_SELECTED_KERNEL_DIR "" CACHE STRING "Kernel source directory selected by kg")
if(KG_SELECTED_KERNEL_DIR STREQUAL "")
  message(FATAL_ERROR "No kernel selected. Use ./kg setup <kernel> or set KG_SELECTED_KERNEL_DIR.")
endif()

cmake_path(ABSOLUTE_PATH KG_SELECTED_KERNEL_DIR BASE_DIRECTORY "${CMAKE_SOURCE_DIR}"
  NORMALIZE OUTPUT_VARIABLE KG_SELECTED_KERNEL_SOURCE)
cmake_path(IS_PREFIX CMAKE_SOURCE_DIR "${KG_SELECTED_KERNEL_SOURCE}" NORMALIZE KG_IN_SOURCE_TREE)
if(NOT KG_IN_SOURCE_TREE OR NOT EXISTS "${KG_SELECTED_KERNEL_SOURCE}/kernel.toml")
  message(FATAL_ERROR "KG_SELECTED_KERNEL_DIR must name a registered kernel directory")
endif()

add_subdirectory("${KG_SELECTED_KERNEL_SOURCE}" "${CMAKE_BINARY_DIR}/kernel")
