#include "render_target.h"

#include "counters.h"

#include <cstdio>

bool RenderTarget::resize(int width, int height) {
    if (width == width_ && height == height_ && fbo_) return true;
    destroy();

    glCreateTextures(GL_TEXTURE_2D, 1, &color_);
    glTextureStorage2D(color_, 1, GL_RGBA8, width, height);
    // Depth as a texture (not a renderbuffer) so Hi-Z culling can sample it later.
    glCreateTextures(GL_TEXTURE_2D, 1, &depth_);
    glTextureStorage2D(depth_, 1, GL_DEPTH_COMPONENT32F, width, height);

    glCreateFramebuffers(1, &fbo_);
    glNamedFramebufferTexture(fbo_, GL_COLOR_ATTACHMENT0, color_, 0);
    glNamedFramebufferTexture(fbo_, GL_DEPTH_ATTACHMENT, depth_, 0);

    width_ = width;
    height_ = height;
    bytes_ = uint64_t(width) * uint64_t(height) * 8; // RGBA8 + D32F
    g_residentBytes += bytes_;

    const GLenum status = glCheckNamedFramebufferStatus(fbo_, GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::fprintf(stderr, "Render target %dx%d incomplete (0x%x)\n", width, height, status);
        return false;
    }
    return true;
}

void RenderTarget::destroy() {
    glDeleteFramebuffers(1, &fbo_);
    glDeleteTextures(1, &color_);
    glDeleteTextures(1, &depth_);
    fbo_ = color_ = depth_ = 0;
    g_residentBytes -= bytes_;
    bytes_ = 0;
    width_ = height_ = 0;
}
