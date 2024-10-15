cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dxgbbench LANGUAGES CXX VERSION 0.0.0)

option(DXGB_USE_CUDA "Enable CUDA implementation" ON)

add_library(dxgbbench SHARED
  ${CMAKE_SOURCE_DIR}/gen.cc
)

if (DXGB_USE_CUDA)
  enable_language(CUDA)
  target_sources(dxgbbench PRIVATE ${CMAKE_SOURCE_DIR}/gen.cu)
  target_compile_options(dxgbbench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)
  target_compile_definitions(dxgbbench PRIVATE -DDXGB_USE_CUDA=1)
endif ()

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
