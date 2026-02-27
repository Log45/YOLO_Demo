cmake_minimum_required(VERSION 3.10)
project(OpenCVDemo)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(ONNX_RUNTIME_PATH "/opt/homebrew/Cellar/onnxruntime/1.24.2")

# Find OpenCV package installed by Homebrew
find_package(OpenCV REQUIRED)

# Add your executable target and link the OpenCV libraries
add_executable(demo main.cpp)
target_include_directories(demo PUBLIC
    "${OpenCV_INCLUDE_DIRS}"
    "${ONNX_RUNTIME_PATH}/include")
target_link_directories(demo PUBLIC
    "${ONNX_RUNTIME_PATH}/lib")
target_link_libraries(demo PUBLIC
    ${OpenCV_LIBS}
    "onnxruntime")