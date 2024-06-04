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
    )

set(SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/logger.cc 
    ${CMAKE_CURRENT_SOURCE_DIR}/exception.cc  
    )

# Create a lib for util module 
add_library(util_module ${SOURCES})

# Include the directory for core module 
target_include_directories(util_module PUBLIC ${HEADERS})

