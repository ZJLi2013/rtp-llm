set(ROCM_DEVICE_SOURCE 
    allocator_rocm.cc 
    )

set(ROCM_DEVICE_HEADER
    allocator_rocm.h 
    hip_helper.h 
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
find_package(hip)

# Create a lib for core module 
add_library(rocm_device_module ${ROCM_DEVICE_SOURCE})

# Include the directory for core module 
target_include_directories(rocm_device_module PUBLIC ${ROCM_DEVICE_HEADER})

# specify dependencies : link against core_module 
target_link_libraries(rocm_device_module PUBLIC core_module)
