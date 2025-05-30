cmake_minimum_required(VERSION 3.11)
project(qllmt LANGUAGES CXX)

set(_saved_CMAKE_MESSAGE_LOG_LEVEL ${CMAKE_MESSAGE_LOG_LEVEL})
set(CMAKE_MESSAGE_LOG_LEVEL ERROR)
add_subdirectory(cutlass)
set(CMAKE_MESSAGE_LOG_LEVEL ${_saved_CMAKE_MESSAGE_LOG_LEVEL})

include_directories("${CMAKE_SOURCE_DIR}")
include_directories(cutlass/tools/util/include)
include_directories(cutlass/include)
include_directories(qllmt/kernels/include)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

# add_subdirectory(qllmt/kernels)
