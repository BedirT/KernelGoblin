cmake_minimum_required(VERSION 3.25)
project(KernelGoblin VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

option(KG_BUILD_TRELLIS2_Z_ORDER "Build the TRELLIS.2 Z-order kernel" OFF)

if(KG_BUILD_TRELLIS2_Z_ORDER)
  add_subdirectory(kernels/trellis2/z_order)
endif()

if(NOT KG_BUILD_TRELLIS2_Z_ORDER)
  message(FATAL_ERROR "No kernels selected. Use ./kg setup <kernel> or a CMake preset.")
endif()
