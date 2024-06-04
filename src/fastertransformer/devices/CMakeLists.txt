

# Specify source files for devices_module 
file(DEVICE_SOURCES  *.cc)

# Specify header files for devices_module 
file(DEVICES_HEADERS *.h)

# Add amd_impl subdirectories  
# add_subdirectory(amd_impl)

# Create a library for devices_module 
add_library(devices_module ${DEVICE_SOURCES})

# Include headers 
target_include_directories(devices_module PUBLIC ${DEVICES_HEADERS})

# specify dependencies
target_link_libraries(devices_module PUBLIC core_module rocm_module utils_module)

