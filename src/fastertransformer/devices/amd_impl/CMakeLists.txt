
set(SOURCE 
    RocmDevice.cc
    )

set(HEADERS
    RocmDevice.h 
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
add_library(amd_impl_module ${SOURCE})

# Include headers 
target_include_directories(amd_impl_module PUBLIC ${HEADERS})

# specify dependencies 
target_link_libraries(amd_impl_module PUBLIC devcies_moduel rocm_module utils_module)