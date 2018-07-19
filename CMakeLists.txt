cmake_minimum_required (VERSION 3.5)

project(UnscentedKF)

set(CMAKE_BUILD_TYPE Debug)
add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

set(sources src/ukf.cpp src/main.cpp src/tools.cpp)
set(headers include/json.hpp include/measurement_package.h include/tools.h include/ukf.h)

#if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# Directories compiler uses to search for include files
include_directories(
  include
  Eigen
  /usr/local/include
  /usr/local/opt/openssl/include
)

link_directories(
  /usr/local/lib
  /usr/local/opt/openssl/lib
  /usr/local/Cellar/libuv/1*/lib
)

#endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

add_executable(UnscentedKF ${sources} ${headers})

target_link_libraries(UnscentedKF z ssl uv uWS)
