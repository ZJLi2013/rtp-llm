message(${CMAKE_CURRENT_SOURCE_DIR})
message(${CMAKE_SOURCE_DIR})

# TODO: 
set(PROJECT_DIR /workspace/)

# Define global include path 
include_directories(${PROJECT_DIR})

set(HEADERS 
    ${CMAKE_CURRENT_SOURCE_DIR}/EnumUtils.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/activation_types.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/RopeTypes.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/layernorm_types.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/logger.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/string_utils.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/exception.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/assert_types.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/ShapeCheck.h 
    )

set(SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/logger.cc 
    ${CMAKE_CURRENT_SOURCE_DIR}/exception.cc  
    ${CMAKE_CURRENT_SOURCE_DIR}/ShapeCheck.cc 
    )

# Configure 
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Build position-independent code (PIC)")

# Create a lib for util module 
add_library(utils_module ${SOURCES})

# Include the directory for core module 
target_include_directories(utils_module PUBLIC ${HEADERS})

# specify dependencies 
target_link_libraries(utils_module PUBLIC core_module)