
set(CORE_SOURCE 
    Types.cc 
    Buffer.cc 
    allocator.cc 
    )

set(CORE_HEADERS
     Types.h 
     Buffer.h 
     allocator.h 
    )


# Create a lib for core module 
add_library(core_module ${CORE_SOURCES})

# Include the directory for core module 
target_include_directories(core_module PUBLIC ${CORE_HEADERS})

# specify dependencies : link against utils_module as core_module depends on utils_module 
target_link_libraries(core_module PUBLIC utils_module)


