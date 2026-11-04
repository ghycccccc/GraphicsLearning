#include "grass_renderer.h"

#include "counters.h"
#include "gl_util.h"
#include "profiler.h"

#include <glm/gtc/packing.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

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
        dst[i] = blade; // whole-struct write: dst may be write-combined mapped memory
    }
}
} // namespace

Animation effectiveAnimation(const GrassRenderSettings& settings) {
    switch (settings.path) {
    case RenderPath::StaticBatch: return Animation::VertexShader;
    case RenderPath::DynamicBatch: return Animation::CpuPerBlade;
    default: return settings.animation;
    }
}

bool usesStreaming(const GrassRenderSettings& settings) {
    if (settings.path == RenderPath::DynamicBatch) return true;
    return effectiveAnimation(settings) == Animation::CpuPerBlade &&
           (settings.path == RenderPath::InstancedAttributes || settings.path == RenderPath::InstancedSsbo);
}

// --- Program -----------------------------------------------------------------

bool GrassRenderer::Program::load(const char* vertexShader, const char* defines) {
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

// --- GrassRenderer -----------------------------------------------------------

bool GrassRenderer::init() {
    if (!uniformProgram_.load("grass.vert", "#define INSTANCE_FROM_UNIFORMS\n") ||
        !attributeProgram_.load("grass.vert", "#define INSTANCE_FROM_ATTRIBUTES\n") ||
        !ssboProgram_.load("grass.vert", "#define INSTANCE_FROM_SSBO\n") ||
        !batchedProgram_.load("grass_batched.vert", "")) {
        return false;
    }

    // Blade mesh only (per-blade draws and the SSBO path).
    glCreateVertexArrays(1, &bladeVao_);
    glEnableVertexArrayAttrib(bladeVao_, 0);
    glVertexArrayAttribFormat(bladeVao_, 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(bladeVao_, 0, 0);

    // Blade mesh on binding 0, per-blade data on binding 1 advancing once per instance.
    glCreateVertexArrays(1, &attributeVao_);
    glEnableVertexArrayAttrib(attributeVao_, 0);
    glVertexArrayAttribFormat(attributeVao_, 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(attributeVao_, 0, 0);
    for (GLuint attrib : {1u, 2u}) {
        glEnableVertexArrayAttrib(attributeVao_, attrib);
        glVertexArrayAttribFormat(attributeVao_, attrib, 4, GL_FLOAT, GL_FALSE, (attrib - 1) * 16);
        glVertexArrayAttribBinding(attributeVao_, attrib, 1);
    }
    glVertexArrayBindingDivisor(attributeVao_, 1, 1);

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

void GrassRenderer::destroy() {
    for (Program* p : {&uniformProgram_, &attributeProgram_, &ssboProgram_, &batchedProgram_})
        glDeleteProgram(p->id);
    glDeleteVertexArrays(1, &bladeVao_);
    glDeleteVertexArrays(1, &attributeVao_);
    glDeleteVertexArrays(1, &batchVao_);
    instanceStream_.release();
    vertexStream_.release();
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
    const bool streaming = usesStreaming(settings);
    if (!streaming || settings.path == RenderPath::DynamicBatch) instanceStream_.release();
    if (settings.path != RenderPath::DynamicBatch) vertexStream_.release();
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
    if (settings.path != RenderPath::PerBladeDraws || effectiveAnimation(settings) != Animation::CpuPerBlade) {
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
    fenceWaitMs_ = 0.0;
    frameBuffer_ = field.instanceBuffer();
    frameOffset_ = 0;
    drawn_ = bladeLimit(field, settings);
    frameBytes_ = uint64_t(drawn_) * sizeof(GrassInstance);
    if (drawn_ == 0) return;

    const bool cpuAnimation = effectiveAnimation(settings) == Animation::CpuPerBlade;
    switch (settings.path) {
    case RenderPath::PerBladeDraws:
        if (cpuAnimation) {
            CpuScope scope(cpu, "CPU animate");
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
        frameBytes_ = uint64_t(drawn_) * uint64_t(verts) * sizeof(BatchedVertex);
        if (!vertexStream_.reserve(frameBytes_, settings.upload)) {
            error_ = "Out of memory for the dynamic batch buffer";
            drawn_ = 0;
            return;
        }
        void* dst = vertexStream_.beginWrite();
        fenceWaitMs_ = vertexStream_.lastWaitMs();
        cpu.record("Fence wait", fenceWaitMs_);
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
        frameOffset_ = vertexStream_.endWrite(frameBytes_);
        frameBuffer_ = vertexStream_.buffer();
        break;
    }

    case RenderPath::InstancedAttributes:
    case RenderPath::InstancedSsbo:
        if (cpuAnimation) {
            if (!instanceStream_.reserve(frameBytes_, settings.upload)) {
                error_ = "Out of memory for the instance upload buffer";
                drawn_ = 0;
                return;
            }
            void* dst = instanceStream_.beginWrite();
            fenceWaitMs_ = instanceStream_.lastWaitMs();
            cpu.record("Fence wait", fenceWaitMs_);
            {
                CpuScope scope(cpu, "CPU animate");
                animateBlades(field.instances().data(), static_cast<GrassInstance*>(dst), drawn_, scene);
            }
            CpuScope scope(cpu, "Upload");
            GpuScope gpuScope(gpu, "Upload");
            frameOffset_ = instanceStream_.endWrite(frameBytes_);
            frameBuffer_ = instanceStream_.buffer();
        }
        break;

    default:
        break;
    }
}

void GrassRenderer::submit(const GrassField& field, const SceneParams& scene,
                           const GrassRenderSettings& settings, CpuProfiler& cpu) {
    if (drawn_ == 0) return;
    CpuScope scope(cpu, "Grass submit");
    const bool cpuAnimation = effectiveAnimation(settings) == Animation::CpuPerBlade;
    const GLsizei bladeIndexCount = GLsizei(field.indicesPerBlade());
    glDisable(GL_CULL_FACE); // blades are two-sided

    switch (settings.path) {
    case RenderPath::PerBladeDraws: {
        const Program& p = uniformProgram_;
        glVertexArrayVertexBuffer(bladeVao_, 0, field.bladeVertexBuffer(), 0, sizeof(glm::vec2));
        glVertexArrayElementBuffer(bladeVao_, field.bladeIndexBuffer());
        glUseProgram(p.id);
        glBindVertexArray(bladeVao_);
        p.setScene(scene, !cpuAnimation);
        g_counters.stateChanges += 2;

        const GrassInstance* blades = cpuAnimation ? animated_.data() : field.instances().data();
        for (int i = 0; i < drawn_; ++i) {
            if (settings.redundantState) {
                // What a naive engine does: every object binds and sets everything it needs.
                glUseProgram(p.id);
                glBindVertexArray(bladeVao_);
                p.setScene(scene, !cpuAnimation);
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
        const GLuint vbo = settings.path == RenderPath::StaticBatch ? staticBatchBuffer_ : frameBuffer_;
        const GLintptr offset = settings.path == RenderPath::StaticBatch ? 0 : frameOffset_;
        glVertexArrayVertexBuffer(batchVao_, 0, vbo, offset, sizeof(BatchedVertex));
        glVertexArrayElementBuffer(batchVao_, batchIndexBuffer_);
        glUseProgram(batchedProgram_.id);
        glBindVertexArray(batchVao_);
        batchedProgram_.setScene(scene, false);
        glDrawElements(GL_TRIANGLES, GLsizei(drawn_) * bladeIndexCount, GL_UNSIGNED_INT, nullptr);
        g_counters.stateChanges += 2;
        g_counters.drawCalls += 1;
        if (settings.path == RenderPath::DynamicBatch) vertexStream_.endFrame();
        break;
    }

    case RenderPath::InstancedAttributes:
        glVertexArrayVertexBuffer(attributeVao_, 0, field.bladeVertexBuffer(), 0, sizeof(glm::vec2));
        glVertexArrayVertexBuffer(attributeVao_, 1, frameBuffer_, frameOffset_, sizeof(GrassInstance));
        glVertexArrayElementBuffer(attributeVao_, field.bladeIndexBuffer());
        glUseProgram(attributeProgram_.id);
        glBindVertexArray(attributeVao_);
        attributeProgram_.setScene(scene, !cpuAnimation);
        glDrawElementsInstanced(GL_TRIANGLES, bladeIndexCount, GL_UNSIGNED_SHORT, nullptr, drawn_);
        g_counters.stateChanges += 2;
        g_counters.drawCalls += 1;
        if (cpuAnimation) instanceStream_.endFrame();
        break;

    case RenderPath::InstancedSsbo:
        glVertexArrayVertexBuffer(bladeVao_, 0, field.bladeVertexBuffer(), 0, sizeof(glm::vec2));
        glVertexArrayElementBuffer(bladeVao_, field.bladeIndexBuffer());
        glUseProgram(ssboProgram_.id);
        glBindVertexArray(bladeVao_);
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, frameBuffer_, frameOffset_, GLsizeiptr(frameBytes_));
        ssboProgram_.setScene(scene, !cpuAnimation);
        glDrawElementsInstanced(GL_TRIANGLES, bladeIndexCount, GL_UNSIGNED_SHORT, nullptr, drawn_);
        g_counters.stateChanges += 3;
        g_counters.drawCalls += 1;
        if (cpuAnimation) instanceStream_.endFrame();
        break;

    default:
        break;
    }
    g_counters.instances += uint64_t(drawn_);
    g_counters.triangles += uint64_t(drawn_) * uint64_t(field.trianglesPerBlade());
}
