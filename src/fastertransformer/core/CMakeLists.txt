

message(${CMAKE_CURRENT_SOURCE_DIR})
message(${CMAKE_SOURCE_DIR})

# TODO: 
set(PROJECT_DIR /workspace/)

# Define global include path 
include_directories(${PROJECT_DIR})

set(SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Types.cc 
    ${CMAKE_CURRENT_SOURCE_DIR}/Buffer.cc 
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cc 
    )

set(HEADERS
     ${CMAKE_CURRENT_SOURCE_DIR}/Types.h 
     ${CMAKE_CURRENT_SOURCE_DIR}/Buffer.h 
     ${CMAKE_CURRENT_SOURCE_DIR}/allocator.h 
    )

# Create a lib for core module 
add_library(core_module ${SOURCES})

# Include the directory for core module 
target_include_directories(core_module PUBLIC ${HEADERS})

# specify dependencies : link against utils_module as core_module depends on utils_module 
target_link_libraries(core_module PUBLIC utils_module)


