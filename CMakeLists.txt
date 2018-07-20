cmake_minimum_required (VERSION 3.5)

project(UnscentedKF)

set(CMAKE_BUILD_TYPE Debug)
add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

# Directories compiler uses to search for include files
include_directories(
  src
  /usr/local/include
  /usr/local/opt/openssl/include
)

link_directories(
  /usr/local/lib
  /usr/local/opt/openssl/lib
  /usr/local/Cellar/libuv/1*/lib
)

add_executable(UnscentedKF
	src/main.cpp
	src/ukf.cpp
	src/tools.cpp
)

target_link_libraries(UnscentedKF z ssl uv uWS)
