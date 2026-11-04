#pragma once
#include <GL/glew.h>

#include <initializer_list>
#include <string>

struct ShaderStage {
    GLenum type;
    const char* file; // relative to SHADER_DIR
};

std::string readTextFile(const std::string& path);

// Compiles and links the given stages. `defines` (e.g. "#define FOO\n") is inserted after
// each stage's #version line. Returns 0 and prints the log on failure.
GLuint loadProgram(std::initializer_list<ShaderStage> stages, const std::string& defines = {});

// Routes KHR_debug messages to stderr (only active with a debug context).
void installDebugOutput();
