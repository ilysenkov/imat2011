cmake_minimum_required(VERSION 2.8)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

set(the_target imat)
project( ${the_target} )

include(OpenCVConfig.cmake)

SET(src_main "main.cpp")
source_group("\\" FILES ${src_main})

file(GLOB src_utils "utils/*.cpp" "utils/*.h*")
source_group("utils" FILES ${src_utils})

file(GLOB src_inc "inc/*.cpp" "inc/*.h*")
source_group("inc" FILES ${src_inc})

file(GLOB src "src/*.cpp")
source_group("src" FILES ${src})

include_directories(${OpenCV_INCLUDE_DIR} ./utils ./src ./inc ${CMAKE_BINARY_DIR})
add_executable( ${the_target} ${src} ${src_utils} ${src_main} ${src_inc})
target_link_libraries( ${the_target} ${OpenCV_LIBS} "-fopenmp")
