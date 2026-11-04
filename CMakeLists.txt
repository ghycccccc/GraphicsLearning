cmake_minimum_required(VERSION 3.24)
project(InstancingLab LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# --- Dependencies ------------------------------------------------------------
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE)

FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
    GIT_SHALLOW TRUE)

# GLEW with pre-generated sources (no Python step needed).
set(glew-cmake_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(glew-cmake_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ONLY_LIBS ON CACHE BOOL "" FORCE)
FetchContent_Declare(glew
    GIT_REPOSITORY https://github.com/Perlmint/glew-cmake.git
    GIT_TAG glew-cmake-2.2.0
    GIT_SHALLOW TRUE)

# Dear ImGui and ImPlot have no CMake build; we compile their sources below.
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.9
    GIT_SHALLOW TRUE)
FetchContent_Declare(implot
    GIT_REPOSITORY https://github.com/epezent/implot.git
    GIT_TAG v0.16
    GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(glfw glm glew imgui implot)
find_package(OpenGL REQUIRED)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends ${implot_SOURCE_DIR})
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

# --- Application -------------------------------------------------------------
function(add_lab_executable name)
    add_executable(${name}
        src/main.cpp
        src/gl_util.cpp
        src/grass.cpp
        src/profiler.cpp
        src/render_target.cpp
        src/grass_renderer.cpp
        src/stream_buffer.cpp)
    target_include_directories(${name} PRIVATE src)
    target_link_libraries(${name} PRIVATE glfw glm::glm libglew_static imgui OpenGL::GL)
    target_compile_definitions(${name} PRIVATE
        GLEW_STATIC
        GLFW_INCLUDE_NONE
        NOMINMAX
        SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders/")
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4 /permissive- /utf-8)
    endif()
endfunction()

add_lab_executable(InstancingLab)
# Same app without the Optimus/PowerXpress export, so laptops run it on the integrated GPU.
add_lab_executable(InstancingLab_iGPU)
target_compile_definitions(InstancingLab_iGPU PRIVATE LAB_INTEGRATED_GPU)

if(MSVC)
    set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT InstancingLab)
endif()
