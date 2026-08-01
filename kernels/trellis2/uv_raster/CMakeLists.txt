if(NOT APPLE)
  message(FATAL_ERROR "trellis2/uv_raster requires Metal on Apple platforms")
endif()

enable_language(OBJCXX)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)

set(KG_METAL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/metal/uv_raster.metal)
set(KG_METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/uv_raster.air)
set(KG_METALLIB ${CMAKE_CURRENT_BINARY_DIR}/uv_raster.metallib)
add_custom_command(
  OUTPUT ${KG_METAL_AIR}
  COMMAND xcrun -sdk macosx metal -std=metal3.0 -O3 -c ${KG_METAL_SOURCE} -o ${KG_METAL_AIR}
  DEPENDS ${KG_METAL_SOURCE}
  COMMENT "Compiling TRELLIS.2 UV raster Metal shader"
  VERBATIM
)
add_custom_command(
  OUTPUT ${KG_METALLIB}
  COMMAND xcrun -sdk macosx metallib ${KG_METAL_AIR} -o ${KG_METALLIB}
  DEPENDS ${KG_METAL_AIR}
  COMMENT "Linking TRELLIS.2 UV raster Metal library"
  VERBATIM
)
add_custom_target(kg_trellis2_uv_raster_metallib DEPENDS ${KG_METALLIB})

add_library(kg_trellis2_uv_raster STATIC src/uv_raster_cpu.cpp src/uv_raster_metal.mm)
add_dependencies(kg_trellis2_uv_raster kg_trellis2_uv_raster_metallib)
target_include_directories(kg_trellis2_uv_raster PUBLIC include)
target_compile_features(kg_trellis2_uv_raster PUBLIC cxx_std_17)
target_compile_options(kg_trellis2_uv_raster PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(kg_trellis2_uv_raster PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_compile_definitions(kg_trellis2_uv_raster PRIVATE KG_METALLIB_PATH="${KG_METALLIB}")
target_link_libraries(kg_trellis2_uv_raster PRIVATE ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK})

add_executable(kg_trellis2_uv_raster_test tests/uv_raster_test.cpp)
target_link_libraries(kg_trellis2_uv_raster_test PRIVATE kg_trellis2_uv_raster)
add_test(NAME trellis2.uv_raster.conformance COMMAND kg_trellis2_uv_raster_test)

add_executable(kg_trellis2_uv_raster_bench bench/uv_raster_bench.cpp)
target_link_libraries(kg_trellis2_uv_raster_bench PRIVATE kg_trellis2_uv_raster)

add_executable(kg_trellis2_uv_raster_cli tools/uv_raster_cli.cpp)
target_link_libraries(kg_trellis2_uv_raster_cli PRIVATE kg_trellis2_uv_raster)
