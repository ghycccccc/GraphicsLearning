#include "gl_util.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

std::string readTextFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "Cannot open %s\n", path.c_str());
        return {};
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static GLuint compileStage(const ShaderStage& stage) {
    const std::string path = std::string(SHADER_DIR) + stage.file;
    const std::string source = readTextFile(path);
    if (source.empty()) return 0;

    GLuint shader = glCreateShader(stage.type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len + 1);
        glGetShaderInfoLog(shader, len, nullptr, log.data());
        std::fprintf(stderr, "Shader compile failed (%s):\n%s\n", stage.file, log.data());
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

GLuint loadProgram(std::initializer_list<ShaderStage> stages) {
    GLuint program = glCreateProgram();
    std::vector<GLuint> shaders;
    bool ok = true;
    for (const ShaderStage& stage : stages) {
        GLuint shader = compileStage(stage);
        if (!shader) {
            ok = false;
            break;
        }
        glAttachShader(program, shader);
        shaders.push_back(shader);
    }

    if (ok) {
        glLinkProgram(program);
        GLint linked = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &linked);
        if (!linked) {
            GLint len = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
            std::vector<char> log(len + 1);
            glGetProgramInfoLog(program, len, nullptr, log.data());
            std::fprintf(stderr, "Program link failed:\n%s\n", log.data());
            ok = false;
        }
    }

    for (GLuint shader : shaders) {
        glDetachShader(program, shader);
        glDeleteShader(shader);
    }
    if (!ok) {
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

static void GLAPIENTRY debugCallback(GLenum, GLenum type, GLuint id, GLenum severity, GLsizei,
                                   const GLchar* message, const void*) {
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;
    std::fprintf(stderr, "[GL %s 0x%x] %s\n", type == GL_DEBUG_TYPE_ERROR ? "ERROR" : "debug", id,
                 message);
}

void installDebugOutput() {
    GLint flags = 0;
    glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    if (!(flags & GL_CONTEXT_FLAG_DEBUG_BIT)) return;
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(debugCallback, nullptr);
}
