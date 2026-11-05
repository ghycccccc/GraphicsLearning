#include "grass_renderer.h"

#include "counters.h"
#include "gl_util.h"
#include "profiler.h"

#include <glm/gtc/packing.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

namespace {
// Vertex memory a batch may use. Static batches live on the GPU; dynamic ones are rewritten
// each frame (and a persistent ring holds three copies).
constexpr uint64_t kStaticBatchBudget = 1024ull << 20;
constexpr uint64_t kDynamicBatchBudget = 256ull << 20;

void animateBlades(const GrassInstance* src, GrassInstance* dst, int count, const SceneParams& scene) {
    for (int i = 0; i < count; ++i) {
        GrassInstance blade = src[i];
        const float sway = windSway({blade.a.x, blade.a.y}, scene.time, scene.windStrength);
        blade.b.z += scene.windDir.x * sway;
        blade.b.w += scene.windDir.y * sway;
        dst[i] = blade;
    }
}
} // namespace

bool isInstanced(RenderPath path) {
    return path == RenderPath::InstancedAttributes || path == RenderPath::InstancedSsbo;
}

bool isSpring(Animation animation) {
    return animation == Animation::SpringCpu || animation == Animation::SpringCpuParallel ||
           animation == Animation::SpringCompute;
}

Animation effectiveAnimation(const GrassRenderSettings& settings) {
    switch (settings.path) {
    case RenderPath::StaticBatch: return Animation::VertexShader;
    case RenderPath::DynamicBatch: return Animation::CpuWind;
    case RenderPath::PerBladeDraws:
        return settings.animation == Animation::CpuWind ? Animation::CpuWind : Animation::VertexShader;
    default: return settings.animation;
    }
}

bool usesStreaming(const GrassRenderSettings& settings) {
    if (settings.path == RenderPath::DynamicBatch) return true;
    const Animation animation = effectiveAnimation(settings);
    return isInstanced(settings.path) && (animation == Animation::CpuWind || animation == Animation::SpringCpu ||
                                          animation == Animation::SpringCpuParallel);
}

// --- Program -----------------------------------------------------------------

bool GrassRenderer::Program::load(const char* vertexShader, const std::string& defines) {
    id = loadProgram({{GL_VERTEX_SHADER, vertexShader}, {GL_FRAGMENT_SHADER, "grass.frag"}}, defines);
    if (!id) return false;
    viewProj = glGetUniformLocation(id, "uViewProj");
    time = glGetUniformLocation(id, "uTime");
    windDir = glGetUniformLocation(id, "uWindDir");
    windStrength = glGetUniformLocation(id, "uWindStrength");
    proceduralWind = glGetUniformLocation(id, "uProceduralWind");
    sunDir = glGetUniformLocation(id, "uSunDir");
    cameraPos = glGetUniformLocation(id, "uCameraPos");
    fogColor = glGetUniformLocation(id, "uFogColor");
    fogDensity = glGetUniformLocation(id, "uFogDensity");
    instanceA = glGetUniformLocation(id, "uInstanceA");
    instanceB = glGetUniformLocation(id, "uInstanceB");
    return true;
}

void GrassRenderer::Program::setScene(const SceneParams& scene, bool procedural) const {
    glProgramUniformMatrix4fv(id, viewProj, 1, GL_FALSE, &scene.viewProj[0][0]);
    glProgramUniform1f(id, time, scene.time);
    glProgramUniform2fv(id, windDir, 1, &scene.windDir[0]);
    glProgramUniform1f(id, windStrength, scene.windStrength);
    glProgramUniform1i(id, proceduralWind, procedural ? 1 : 0);
    glProgramUniform3fv(id, sunDir, 1, &scene.sunDir[0]);
    glProgramUniform3fv(id, cameraPos, 1, &scene.cameraPos[0]);
    glProgramUniform3fv(id, fogColor, 1, &scene.fogColor[0]);
    glProgramUniform1f(id, fogDensity, scene.fogDensity);
    g_counters.uniformUpdates += 9;
}

const GrassRenderer::Program& GrassRenderer::grassProgram(const std::string& defines) {
    auto it = grassPrograms_.find(defines);
    if (it == grassPrograms_.end()) {
        Program program;
        program.load("grass.vert", defines); // a failed compile leaves id 0: nothing draws
        it = grassPrograms_.emplace(defines, program).first;
    }
    return it->second;
}

// --- GrassRenderer -----------------------------------------------------------

bool GrassRenderer::init() {
    // Compile the variants used at startup now, so shader errors surface immediately.
    if (!grassProgram("#define INSTANCE_FROM_UNIFORMS\n").id ||
        !grassProgram("#define INSTANCE_FROM_SSBO\n").id ||
        !batchedProgram_.load("grass_batched.vert", "") || !sim_.init()) {
        return false;
    }

    // Blade mesh only (per-blade draws and the SSBO path).
    glCreateVertexArrays(1, &bladeVao_);
    glEnableVertexArrayAttrib(bladeVao_, 0);
    glVertexArrayAttribFormat(bladeVao_, 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(bladeVao_, 0, 0);

    // Blade mesh on binding 0, static blade data on binding 1 and bends on binding 2, both
    // advancing once per instance. Attribute formats depend on the layout.
    glCreateVertexArrays(1, &attributeVao_);
    glEnableVertexArrayAttrib(attributeVao_, 0);
    glVertexArrayAttribFormat(attributeVao_, 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(attributeVao_, 0, 0);
    for (GLuint attrib = 1; attrib <= 5; ++attrib) {
        glEnableVertexArrayAttrib(attributeVao_, attrib);
        glVertexArrayAttribBinding(attributeVao_, attrib, 1);
    }
    glVertexArrayAttribBinding(attributeVao_, 6, 2);
    glVertexArrayBindingDivisor(attributeVao_, 1, 1);
    glVertexArrayBindingDivisor(attributeVao_, 2, 1);
    configureAttributeVao(InstanceLayout::Float32);

    // Pre-transformed batch vertices.
    glCreateVertexArrays(1, &batchVao_);
    glEnableVertexArrayAttrib(batchVao_, 0);
    glVertexArrayAttribFormat(batchVao_, 0, 4, GL_FLOAT, GL_FALSE, offsetof(BatchedVertex, posT));
    glEnableVertexArrayAttrib(batchVao_, 1);
    glVertexArrayAttribFormat(batchVao_, 1, 4, GL_INT_2_10_10_10_REV, GL_TRUE, offsetof(BatchedVertex, normal));
    glEnableVertexArrayAttrib(batchVao_, 2);
    glVertexArrayAttribFormat(batchVao_, 2, 3, GL_FLOAT, GL_FALSE, offsetof(BatchedVertex, variation));
    for (GLuint attrib : {0u, 1u, 2u}) glVertexArrayAttribBinding(batchVao_, attrib, 0);
    return true;
}

void GrassRenderer::configureAttributeVao(InstanceLayout layout) {
    if (layout == attributeVaoLayout_) return;
    attributeVaoLayout_ = layout;
    // The vertex fetch does the unpacking for free: the shader sees the same floats either way.
    if (layout == InstanceLayout::Float32) {
        glVertexArrayAttribFormat(attributeVao_, 1, 2, GL_FLOAT, GL_FALSE, 0);  // root
        glVertexArrayAttribFormat(attributeVao_, 2, 1, GL_FLOAT, GL_FALSE, 8);  // yaw / 2pi
        glVertexArrayAttribFormat(attributeVao_, 3, 1, GL_FLOAT, GL_FALSE, 12); // variation
        glVertexArrayAttribFormat(attributeVao_, 4, 1, GL_FLOAT, GL_FALSE, 24); // rest bend / 0.5
        glVertexArrayAttribFormat(attributeVao_, 5, 2, GL_FLOAT, GL_FALSE, 16); // height, width
        glVertexArrayAttribFormat(attributeVao_, 6, 2, GL_FLOAT, GL_FALSE, 0);  // bend
    } else {
        glVertexArrayAttribFormat(attributeVao_, 1, 2, GL_FLOAT, GL_FALSE, 0);
        glVertexArrayAttribFormat(attributeVao_, 2, 1, GL_UNSIGNED_SHORT, GL_TRUE, 8);
        glVertexArrayAttribFormat(attributeVao_, 3, 1, GL_UNSIGNED_BYTE, GL_TRUE, 10);
        glVertexArrayAttribFormat(attributeVao_, 4, 1, GL_UNSIGNED_BYTE, GL_TRUE, 11);
        glVertexArrayAttribFormat(attributeVao_, 5, 2, GL_HALF_FLOAT, GL_FALSE, 12);
        glVertexArrayAttribFormat(attributeVao_, 6, 2, GL_HALF_FLOAT, GL_FALSE, 0);
    }
}

void GrassRenderer::destroy() {
    for (auto& [defines, program] : grassPrograms_) glDeleteProgram(program.id);
    glDeleteProgram(batchedProgram_.id);
    glDeleteVertexArrays(1, &bladeVao_);
    glDeleteVertexArrays(1, &attributeVao_);
    glDeleteVertexArrays(1, &batchVao_);
    staticStream_.release();
    bendStream_.release();
    vertexStream_.release();
    sim_.destroy();
    glDeleteBuffers(1, &batchIndexBuffer_);
    glDeleteBuffers(1, &staticBatchBuffer_);
}

int GrassRenderer::bladeLimit(const GrassField& field, const GrassRenderSettings& settings) const {
    const int count = field.count();
    switch (settings.path) {
    case RenderPath::PerBladeDraws:
        return std::min(count, settings.cpuPathCap);
    case RenderPath::StaticBatch:
    case RenderPath::DynamicBatch: {
        const uint64_t budget = settings.path == RenderPath::StaticBatch ? kStaticBatchBudget : kDynamicBatchBudget;
        const uint64_t perBlade = uint64_t(field.verticesPerBlade()) * sizeof(BatchedVertex);
        return int(std::min<uint64_t>({uint64_t(count), uint64_t(settings.cpuPathCap), budget / perBlade}));
    }
    default:
        return count;
    }
}

void GrassRenderer::releaseUnused(const GrassRenderSettings& settings) {
    const Animation animation = effectiveAnimation(settings);
    const bool instancedStreaming = isInstanced(settings.path) && usesStreaming(settings);
    if (!instancedStreaming) bendStream_.release();
    if (!instancedStreaming || settings.uploadScope != UploadScope::Full) staticStream_.release();
    if (settings.path != RenderPath::DynamicBatch) vertexStream_.release();
    if (!isInstanced(settings.path) || !isSpring(animation)) sim_.reset();
    if (settings.path != RenderPath::StaticBatch && staticBatchBuffer_) {
        glDeleteBuffers(1, &staticBatchBuffer_);
        staticBatchBuffer_ = 0;
        g_residentBytes -= staticBatchBytes_;
        staticBatchBytes_ = 0;
        staticBatchVersion_ = ~0ull;
    }
    if (settings.path != RenderPath::StaticBatch && settings.path != RenderPath::DynamicBatch &&
        batchIndexBuffer_) {
        glDeleteBuffers(1, &batchIndexBuffer_);
        batchIndexBuffer_ = 0;
        g_residentBytes -= batchIndexBytes_;
        batchIndexBytes_ = 0;
        batchIndexVersion_ = ~0ull;
    }
    if (settings.path != RenderPath::PerBladeDraws || animation != Animation::CpuWind) {
        animated_.clear();
        animated_.shrink_to_fit();
    }
}

void GrassRenderer::writeBladeVertices(const GrassField& field, const GrassInstance& blade, glm::vec2 bend,
                                       bool proceduralWind, BatchedVertex* out) {
    // Same math as grass.vert, done on the CPU once per vertex.
    const glm::vec3 root(blade.a.x, 0.0f, blade.a.y);
    const glm::vec3 right(std::cos(blade.a.z), 0.0f, -std::sin(blade.a.z));
    const float h = blade.b.x, w = blade.b.y;
    const glm::vec3 bend3(bend.x, 0.0f, bend.y);
    const float phase = blade.a.x * 0.13f + blade.a.y * 0.09f;
    for (const glm::vec2& v : field.bladeVertices()) {
        const float t = v.y;
        const glm::vec3 p = root + right * (v.x * w * (1.0f - t)) + glm::vec3(0.0f, t * h, 0.0f) + bend3 * (h * t * t);
        const glm::vec3 dPdt = glm::vec3(0.0f, h, 0.0f) + bend3 * (2.0f * h * t);
        const glm::vec3 n = glm::normalize(glm::cross(right, dPdt));
        *out++ = {glm::vec4(p, t), glm::packSnorm3x10_1x2(glm::vec4(n, 0.0f)), blade.a.w, phase,
                  proceduralWind ? h * t * t : 0.0f};
    }
}

void GrassRenderer::ensureBatchIndices(const GrassField& field, int blades) {
    if (batchIndexBuffer_ && batchIndexVersion_ == field.version() && batchIndexBlades_ == blades) return;
    const std::vector<uint16_t>& blade = field.bladeIndices();
    const uint32_t verts = uint32_t(field.verticesPerBlade());
    std::vector<uint32_t> indices;
    indices.reserve(size_t(blades) * blade.size());
    for (uint32_t b = 0; b < uint32_t(blades); ++b)
        for (uint16_t i : blade) indices.push_back(b * verts + i);

    glDeleteBuffers(1, &batchIndexBuffer_);
    g_residentBytes -= batchIndexBytes_;
    batchIndexBytes_ = indices.size() * sizeof(uint32_t);
    glCreateBuffers(1, &batchIndexBuffer_);
    glNamedBufferStorage(batchIndexBuffer_, GLsizeiptr(std::max<uint64_t>(batchIndexBytes_, 4)), indices.data(), 0);
    g_residentBytes += batchIndexBytes_;
    g_counters.bytesUploaded += batchIndexBytes_;
    batchIndexVersion_ = field.version();
    batchIndexBlades_ = blades;
}

void GrassRenderer::ensureStaticBatch(const GrassField& field, int blades, CpuProfiler& cpu) {
    if (staticBatchBuffer_ && staticBatchVersion_ == field.version() && staticBatchBlades_ == blades) return;
    CpuScope scope(cpu, "Build static batch");
    const int verts = field.verticesPerBlade();
    std::vector<BatchedVertex> vertices(size_t(blades) * size_t(verts));
    for (int b = 0; b < blades; ++b) {
        const GrassInstance& blade = field.instances()[size_t(b)];
        writeBladeVertices(field, blade, {blade.b.z, blade.b.w}, true, &vertices[size_t(b) * size_t(verts)]);
    }
    glDeleteBuffers(1, &staticBatchBuffer_);
    g_residentBytes -= staticBatchBytes_;
    staticBatchBytes_ = vertices.size() * sizeof(BatchedVertex);
    glCreateBuffers(1, &staticBatchBuffer_);
    glNamedBufferStorage(staticBatchBuffer_, GLsizeiptr(std::max<uint64_t>(staticBatchBytes_, 4)), vertices.data(), 0);
    g_residentBytes += staticBatchBytes_;
    g_counters.bytesUploaded += staticBatchBytes_;
    staticBatchVersion_ = field.version();
    staticBatchBlades_ = blades;
}

void GrassRenderer::prepare(const GrassField& field, const SceneParams& scene,
                            const GrassRenderSettings& settings, CpuProfiler& cpu, GpuProfiler& gpu) {
    releaseUnused(settings);
    error_ = nullptr;
    drawn_ = bladeLimit(field, settings);
    if (drawn_ == 0) return;

    switch (settings.path) {
    case RenderPath::PerBladeDraws:
        if (effectiveAnimation(settings) == Animation::CpuWind) {
            CpuScope scope(cpu, "CPU wind");
            animated_.resize(size_t(drawn_));
            animateBlades(field.instances().data(), animated_.data(), drawn_, scene);
        }
        break;

    case RenderPath::StaticBatch:
        ensureBatchIndices(field, drawn_);
        ensureStaticBatch(field, drawn_, cpu);
        break;

    case RenderPath::DynamicBatch: {
        ensureBatchIndices(field, drawn_);
        const int verts = field.verticesPerBlade();
        const uint64_t bytes = uint64_t(drawn_) * uint64_t(verts) * sizeof(BatchedVertex);
        if (!vertexStream_.reserve(bytes, settings.upload)) {
            error_ = "Out of memory for the dynamic batch buffer";
            drawn_ = 0;
            return;
        }
        void* dst = vertexStream_.beginWrite();
        cpu.record("Fence wait", vertexStream_.lastWaitMs());
        {
            CpuScope scope(cpu, "CPU build vertices");
            auto* out = static_cast<BatchedVertex*>(dst);
            for (int b = 0; b < drawn_; ++b) {
                const GrassInstance& blade = field.instances()[size_t(b)];
                const float sway = windSway({blade.a.x, blade.a.y}, scene.time, scene.windStrength);
                writeBladeVertices(field, blade, glm::vec2(blade.b.z, blade.b.w) + scene.windDir * sway, false,
                                   out + size_t(b) * size_t(verts));
            }
        }
        CpuScope scope(cpu, "Upload");
        GpuScope gpuScope(gpu, "Upload");
        frameVertexOffset_ = vertexStream_.endWrite(bytes);
        frameVertexBuffer_ = vertexStream_.buffer();
        break;
    }

    case RenderPath::InstancedAttributes:
    case RenderPath::InstancedSsbo:
        prepareInstanced(field, scene, settings, cpu, gpu);
        break;

    default:
        break;
    }
}

void GrassRenderer::prepareInstanced(const GrassField& field, const SceneParams& scene,
                                     const GrassRenderSettings& settings, CpuProfiler& cpu, GpuProfiler& gpu) {
    const Animation animation = effectiveAnimation(settings);
    const InstanceLayout layout = field.layout();
    frameStaticBuffer_ = field.staticBuffer();
    frameStaticOffset_ = 0;
    frameBendBuffer_ = 0;
    frameBendOffset_ = 0;
    frameDynamicBend_ = animation != Animation::VertexShader;

    SimParams sim;
    sim.time = scene.time;
    sim.stiffness = settings.stiffness;
    sim.damping = settings.damping;
    sim.windDir = scene.windDir;
    sim.windStrength = scene.windStrength;
    sim.trampleCenter = scene.trampleCenter;
    sim.trampleRadius = scene.trampleRadius;

    if (animation == Animation::SpringCompute) {
        sim_.stepGpu(field, sim, settings.readback, cpu, gpu);
        frameBendBuffer_ = sim_.bendBuffer();
        return;
    }
    if (!usesStreaming(settings)) return;

    // CPU-owned bends: write them straight into this frame's upload region.
    const uint64_t bendBytes = uint64_t(drawn_) * bendStride(layout);
    const bool full = settings.uploadScope == UploadScope::Full;
    const uint64_t staticBytes = full ? uint64_t(drawn_) * staticStride(layout) : 0;
    if (!bendStream_.reserve(bendBytes, settings.upload) ||
        (full && !staticStream_.reserve(staticBytes, settings.upload))) {
        error_ = "Out of memory for the upload buffers";
        drawn_ = 0;
        return;
    }
    void* bends = bendStream_.beginWrite();
    cpu.record("Fence wait", bendStream_.lastWaitMs());
    if (animation == Animation::CpuWind) {
        CpuScope scope(cpu, "CPU wind");
        for (int i = 0; i < drawn_; ++i) {
            const GrassInstance& blade = field.instances()[size_t(i)];
            const float sway = windSway({blade.a.x, blade.a.y}, scene.time, scene.windStrength);
            storeBend(bends, size_t(i), glm::vec2(blade.b.z, blade.b.w) + scene.windDir * sway, layout);
        }
    } else {
        sim_.stepCpu(field, sim, animation == Animation::SpringCpuParallel, bends, cpu);
    }

    CpuScope scope(cpu, "Upload");
    GpuScope gpuScope(gpu, "Upload");
    if (full) {
        // An engine that re-sends whole instance records: copy the unchanged static data too.
        std::memcpy(staticStream_.beginWrite(), field.staticBytes().data(), size_t(staticBytes));
        frameStaticOffset_ = staticStream_.endWrite(staticBytes);
        frameStaticBuffer_ = staticStream_.buffer();
    }
    frameBendOffset_ = bendStream_.endWrite(bendBytes);
    frameBendBuffer_ = bendStream_.buffer();
}

void GrassRenderer::submit(const GrassField& field, const SceneParams& scene,
                           const GrassRenderSettings& settings, CpuProfiler& cpu) {
    if (drawn_ == 0) return;
    CpuScope scope(cpu, "Grass submit");
    const Animation animation = effectiveAnimation(settings);
    const GLsizei bladeIndexCount = GLsizei(field.indicesPerBlade());
    glDisable(GL_CULL_FACE); // blades are two-sided

    switch (settings.path) {
    case RenderPath::PerBladeDraws: {
        const Program& p = grassProgram("#define INSTANCE_FROM_UNIFORMS\n");
        const bool cpuWind = animation == Animation::CpuWind;
        glVertexArrayVertexBuffer(bladeVao_, 0, field.bladeVertexBuffer(), 0, sizeof(glm::vec2));
        glVertexArrayElementBuffer(bladeVao_, field.bladeIndexBuffer());
        glUseProgram(p.id);
        glBindVertexArray(bladeVao_);
        p.setScene(scene, !cpuWind);
        g_counters.stateChanges += 2;

        const GrassInstance* blades = cpuWind ? animated_.data() : field.instances().data();
        for (int i = 0; i < drawn_; ++i) {
            if (settings.redundantState) {
                // What a naive engine does: every object binds and sets everything it needs.
                glUseProgram(p.id);
                glBindVertexArray(bladeVao_);
                p.setScene(scene, !cpuWind);
                g_counters.stateChanges += 2;
            }
            glUniform4fv(p.instanceA, 1, &blades[i].a[0]);
            glUniform4fv(p.instanceB, 1, &blades[i].b[0]);
            glDrawElements(GL_TRIANGLES, bladeIndexCount, GL_UNSIGNED_SHORT, nullptr);
        }
        g_counters.uniformUpdates += uint64_t(drawn_) * 2;
        g_counters.drawCalls += uint64_t(drawn_);
        break;
    }

    case RenderPath::StaticBatch:
    case RenderPath::DynamicBatch: {
        const bool isStatic = settings.path == RenderPath::StaticBatch;
        glVertexArrayVertexBuffer(batchVao_, 0, isStatic ? staticBatchBuffer_ : frameVertexBuffer_,
                                  isStatic ? 0 : frameVertexOffset_, sizeof(BatchedVertex));
        glVertexArrayElementBuffer(batchVao_, batchIndexBuffer_);
        glUseProgram(batchedProgram_.id);
        glBindVertexArray(batchVao_);
        batchedProgram_.setScene(scene, false);
        glDrawElements(GL_TRIANGLES, GLsizei(drawn_) * bladeIndexCount, GL_UNSIGNED_INT, nullptr);
        g_counters.stateChanges += 2;
        g_counters.drawCalls += 1;
        if (!isStatic) vertexStream_.endFrame();
        break;
    }

    case RenderPath::InstancedAttributes:
    case RenderPath::InstancedSsbo: {
        const InstanceLayout layout = field.layout();
        const bool ssbo = settings.path == RenderPath::InstancedSsbo;
        std::string defines = ssbo ? "#define INSTANCE_FROM_SSBO\n" : "#define INSTANCE_FROM_ATTRIBUTES\n";
        if (ssbo && layout == InstanceLayout::Packed16) defines += "#define PACKED_LAYOUT\n";
        if (frameDynamicBend_) defines += "#define DYNAMIC_BEND\n";
        const Program& p = grassProgram(defines);

        if (ssbo) {
            glVertexArrayVertexBuffer(bladeVao_, 0, field.bladeVertexBuffer(), 0, sizeof(glm::vec2));
            glVertexArrayElementBuffer(bladeVao_, field.bladeIndexBuffer());
            glBindVertexArray(bladeVao_);
            glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, frameStaticBuffer_, frameStaticOffset_,
                              GLsizeiptr(uint64_t(drawn_) * staticStride(layout)));
            if (frameDynamicBend_) {
                glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, frameBendBuffer_, frameBendOffset_,
                                  GLsizeiptr(uint64_t(drawn_) * bendStride(layout)));
            }
        } else {
            configureAttributeVao(layout);
            glVertexArrayVertexBuffer(attributeVao_, 0, field.bladeVertexBuffer(), 0, sizeof(glm::vec2));
            glVertexArrayVertexBuffer(attributeVao_, 1, frameStaticBuffer_, frameStaticOffset_, staticStride(layout));
            // An enabled array needs a buffer, so the bend attribute is only on when used.
            if (frameDynamicBend_) {
                glEnableVertexArrayAttrib(attributeVao_, 6);
                glVertexArrayVertexBuffer(attributeVao_, 2, frameBendBuffer_, frameBendOffset_, bendStride(layout));
            } else {
                glDisableVertexArrayAttrib(attributeVao_, 6);
            }
            glVertexArrayElementBuffer(attributeVao_, field.bladeIndexBuffer());
            glBindVertexArray(attributeVao_);
        }
        glUseProgram(p.id);
        p.setScene(scene, animation == Animation::VertexShader);
        glDrawElementsInstanced(GL_TRIANGLES, bladeIndexCount, GL_UNSIGNED_SHORT, nullptr, drawn_);
        g_counters.stateChanges += frameDynamicBend_ ? 4 : 3;
        g_counters.drawCalls += 1;
        if (usesStreaming(settings)) {
            bendStream_.endFrame();
            if (settings.uploadScope == UploadScope::Full) staticStream_.endFrame();
        }
        break;
    }

    default:
        break;
    }
    g_counters.instances += uint64_t(drawn_);
    g_counters.triangles += uint64_t(drawn_) * uint64_t(field.trianglesPerBlade());
}
