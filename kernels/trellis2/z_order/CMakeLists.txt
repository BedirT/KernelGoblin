if(NOT APPLE)
  message(FATAL_ERROR "trellis2/z_order currently provides a Metal backend and requires Apple platforms")
endif()

enable_language(OBJCXX)

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)

set(KG_METAL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/metal/z_order.metal)
set(KG_METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/z_order.air)
set(KG_METALLIB ${CMAKE_CURRENT_BINARY_DIR}/z_order.metallib)

add_custom_command(
  OUTPUT ${KG_METAL_AIR}
  COMMAND xcrun -sdk macosx metal -std=metal3.0 -O3 -c ${KG_METAL_SOURCE} -o ${KG_METAL_AIR}
  DEPENDS ${KG_METAL_SOURCE}
  COMMENT "Compiling TRELLIS.2 Z-order Metal shader"
  VERBATIM
)
add_custom_command(
  OUTPUT ${KG_METALLIB}
  COMMAND xcrun -sdk macosx metallib ${KG_METAL_AIR} -o ${KG_METALLIB}
  DEPENDS ${KG_METAL_AIR}
  COMMENT "Linking TRELLIS.2 Z-order Metal library"
  VERBATIM
)
add_custom_target(kg_trellis2_z_order_metallib DEPENDS ${KG_METALLIB})

add_library(kg_trellis2_z_order STATIC
  src/z_order_cpu.cpp
  src/z_order_metal.mm
)
add_dependencies(kg_trellis2_z_order kg_trellis2_z_order_metallib)
target_include_directories(kg_trellis2_z_order PUBLIC include)
target_compile_features(kg_trellis2_z_order PUBLIC cxx_std_17)
target_compile_options(kg_trellis2_z_order PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(kg_trellis2_z_order PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_compile_definitions(kg_trellis2_z_order PRIVATE KG_METALLIB_PATH="${KG_METALLIB}")
target_link_libraries(kg_trellis2_z_order PRIVATE ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK})

add_executable(kg_trellis2_z_order_test tests/z_order_test.cpp)
target_link_libraries(kg_trellis2_z_order_test PRIVATE kg_trellis2_z_order)
add_test(NAME trellis2.z_order.conformance COMMAND kg_trellis2_z_order_test)

add_executable(kg_trellis2_z_order_bench bench/z_order_bench.cpp)
target_link_libraries(kg_trellis2_z_order_bench PRIVATE kg_trellis2_z_order)
