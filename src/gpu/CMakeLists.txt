# GPU support for MOPAC (CUDA + cuBLAS + optional MAGMA)
#
# This directory contains:
#   - Fortran interface modules (iso_c_binding) for calling CUDA/cuBLAS from Fortran
#   - CUDA source files (.cu) with GPU kernels and cuBLAS driver wrappers

#===============================================
# Fortran interface modules for GPU
set(gpu_fortran_src
    mod_vars_cuda
    mod_gpu_info
    mod_calls_cublas
    density_cuda_i
    mamult_cuda_i
    pulay_for_gpu
)

foreach(idx IN LISTS gpu_fortran_src)
    target_sources(mopac-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${idx}.F90)
endforeach()

#===============================================
# CUDA source files
set(gpu_cuda_src
    ${CMAKE_CURRENT_SOURCE_DIR}/cublas_drivers.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/diag_density_cuda.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_info.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/mamult_cuda.cu
)

# Create a static CUDA library and link to mopac-core
add_library(mopac-gpu STATIC ${gpu_cuda_src})
set_target_properties(mopac-gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(mopac-gpu PUBLIC CUDA::cudart CUDA::cublas)

if(MAGMA)
    target_compile_definitions(mopac-gpu PRIVATE MAGMA)
    target_link_libraries(mopac-gpu PUBLIC ${MAGMA_LIBRARY})
    target_include_directories(mopac-gpu PUBLIC ${MAGMA_INCLUDE})
endif()

target_link_libraries(mopac-core PUBLIC mopac-gpu)
#===============================================
