

# TODO: 
set(PROJECT_DIR /workspace/)

# TODO: 指定全局包含目录 for rtp-llm project 
include_directories(${PROJECT_DIR})

set(SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator_rocm.cc 
    )

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator_rocm.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/hip_helper.h 
    )

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

# Set HIPCC compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

# Create a lib for core module 
add_library(rocm_device_module ${SOURCES})

# Include the directory for core module 
target_include_directories(rocm_device_module PUBLIC 
                            ${HEADERS}
                            ${ROCM_PATH}/include )

# specify dependencies : link against core_module 
target_link_libraries(rocm_device_module PUBLIC core_module hip::host )
