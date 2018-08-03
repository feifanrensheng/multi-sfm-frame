cmake_minimum_required( VERSION 2.8 )
project( sfm )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(OpenCV_DIR "/usr/local/opencv-3.1.0/share/OpenCV")
include_directories("/usr/local/opencv-3.1.0/include")

# opencv 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# eigen 
include_directories( "/usr/include/eigen3/" )

# pcl
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# opengl
find_package(OpenGL REQUIRED )
include_directories(${OpenGL_INCLUDE_DIR})
link_directories(${OpenGL_LIBRARY_DIR})

#glut
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
add_definitions(${GLUT_DEFINITIONS})


add_executable( test-sfm test-main.cpp ARDrawingContext.cpp ARDrawingContext.hpp CameraCalibration.cpp CameraCalibration.hpp GeometryTypes.cpp GeometryTypes.hpp)
target_link_libraries( test-sfm ${OpenCV_LIBS} ${PCL_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

add_executable( test test.cpp )
target_link_libraries( test ${OpenCV_LIBS} ${PCL_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
