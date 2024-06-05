

# TODO: 
set(PROJECT_DIR /workspace/)
# Define global include path 
include_directories(${PROJECT_DIR})

# Specify source files for devices_module 
file(GLOB SOURCES *.cc)
message({SOURCES})

# Specify header files for devices_module 
file(GLOB HEADERS *.h)
message({HEADERS})

# Add amd_impl subdirectories  
add_subdirectory(amd_impl)

# Create a library for devices_module 
add_library(devices_module SHARED ${SOURCES})

# Include headers 
target_include_directories(devices_module PUBLIC ${HEADERS})

# specify dependencies
target_link_libraries(devices_module PUBLIC core_module util_module)

