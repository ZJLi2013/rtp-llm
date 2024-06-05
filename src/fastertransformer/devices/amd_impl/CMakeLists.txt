
# TODO: 
set(PROJECT_DIR /workspace/)
# Define global include path 
include_directories(${PROJECT_DIR})

set(SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/RocmDevice.cc
    )

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/RocmDevice.h 
    )

# Configure 
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Build position-independent code (PIC)")

if(${__HIP_ENABLE_PCH})
  add_definitions(-D__HIP_ENABLE_PCH)
endif()

if(UNIX)
  if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
  endif()
  # Search for rocm in common locations
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()

# Find hip
find_package(hip REQUIRED)

# Find hipblas rocm_smi libraries 
find_library(HIPBLAS hipblas HINTS /opt/rocm/lib)
find_library(HIPBLASLT hipblaslt HINTS /opt/rocm/lib)
find_library(RMSI rocm_smi64 HINTS /opt/rocm/lib)

# Set HIPCC compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

# Create a lib for core module 
add_library(amd_impl_module ${SOURCES})

# Include headers 
target_include_directories(amd_impl_module PUBLIC ${HEADERS} ${ROCM_PATH}/include)

# specify dependencies 
target_link_libraries(amd_impl_module PUBLIC devices_module rocm_module utils_module hip::host ${HIPBLAS} ${HIPBLASLT} ${RMSI})

