cmake_minimum_required(VERSION 3.20)
project(untitled)

set(CMAKE_CXX_STANDARD 20)

set(IMGUI_SOURCES
        imgui/imgui.cpp
        imgui/imgui_demo.cpp
        imgui/imgui_draw.cpp
        imgui/imgui_tables.cpp
        imgui/imgui_widgets.cpp
        imgui/imgui_impl_glfw.cpp
        imgui/imgui_impl_opengl3.cpp
)

add_executable(untitled
        main.cpp
        Glad/glad.c
        ${IMGUI_SOURCES}
        matrix.h
        mnist.h
        network.h
)

target_include_directories(untitled PRIVATE
        ${CMAKE_SOURCE_DIR}/Dependencies/include
        ${CMAKE_SOURCE_DIR}/imgui
)

target_link_directories(untitled PRIVATE ${CMAKE_SOURCE_DIR}/Dependencies/lib)
target_link_libraries(untitled PRIVATE glfw3 opengl32)