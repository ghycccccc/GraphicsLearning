#pragma once
#include <GL/glew.h>

#include <cstdint>

// Offscreen color + depth target. The scene renders here at (window size x render scale)
// and is then blitted to the window, so fill-rate cost can be dialed independently.
class RenderTarget {
public:
    // Reallocates only when the size changes. Returns false if the framebuffer is incomplete.
    bool resize(int width, int height);
    void destroy();

    GLuint framebuffer() const { return fbo_; }
    GLuint depthTexture() const { return depth_; }
    int width() const { return width_; }
    int height() const { return height_; }

private:
    GLuint fbo_ = 0, color_ = 0, depth_ = 0;
    int width_ = 0, height_ = 0;
    uint64_t bytes_ = 0;
};
