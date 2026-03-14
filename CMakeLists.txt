cmake_minimum_required(VERSION 3.15)
project(lept_skew C)

find_package(Leptonica CONFIG REQUIRED)

add_executable(lept_skew lept_skew.c)
target_include_directories(lept_skew PRIVATE ${Leptonica_INCLUDE_DIRS})
target_link_libraries(lept_skew PRIVATE ${Leptonica_LIBRARIES})
