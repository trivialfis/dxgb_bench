cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dxgbbench LANGUAGES CXX VERSION 0.0.0)

option(DXGB_USE_CUDA "Enable CUDA implementation" ON)

add_library(dxgbbench SHARED ${CMAKE_SOURCE_DIR}/gen.cc)

if (DXGB_USE_CUDA)
  # Require thrust omp implementation.
  # cc version produces different result: https://github.com/NVIDIA/cccl/issues/2602
  find_package(OpenMP REQUIRED)
  target_link_libraries(dxgbbench PRIVATE OpenMP::OpenMP_CXX)

  enable_language(CUDA)
  target_sources(dxgbbench PRIVATE ${CMAKE_SOURCE_DIR}/gen.cu)
  find_package(CUDAToolkit REQUIRED)
  find_package(CCCL CONFIG HINTS ${CUDAToolkit_LIBRARY_DIR}/cmake)
  if(NOT CCCL_FOUND)
    message(FATAL_ERROR "Standalone CCCL not found. Attempting to use CCCL from CUDA Toolkit...")
  endif()
  target_link_libraries(dxgbbench PRIVATE CUDA::cudart CCCL::CCCL)
  target_compile_options(dxgbbench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
  set_target_properties(dxgbbench PROPERTIES
    CUDA_VISIBILITY_PRESET hidden
  )
  target_compile_definitions(dxgbbench PRIVATE -DDXGB_USE_CUDA=1)
  target_compile_options(dxgbbench PRIVATE )
endif ()

set_target_properties(dxgbbench PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
)

function(set_output_directory target dir)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${dir}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${dir}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${dir}
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${dir}
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${dir}
    LIBRARY_OUTPUT_DIRECTORY ${dir}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${dir}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${dir}
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${dir}
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${dir}
    ARCHIVE_OUTPUT_DIRECTORY ${dir}
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${dir}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${dir}
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${dir}
    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${dir})
endfunction()

set_output_directory(dxgbbench ${CMAKE_SOURCE_DIR})
