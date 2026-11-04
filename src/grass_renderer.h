#pragma once
#include "grass.h"
#include "stream_buffer.h"

#include <cstdint>
#include <vector>

class CpuProfiler;
class GpuProfiler;

enum class RenderPath {
    PerBladeDraws,       // A0: one glDrawElements per blade, data as uniforms
    StaticBatch,         // A1: all blades pre-transformed into one big mesh, built once
    DynamicBatch,        // A2: the CPU re-transforms every vertex each frame
    InstancedAttributes, // A3: one instanced draw, per-blade data as divisor-1 attributes
    InstancedSsbo,       // A3: one instanced draw, per-blade data fetched from an SSBO
    kCount
};

enum class Animation {
    VertexShader, // per-blade data is static on the GPU; wind computed in the vertex shader
    CpuPerBlade,  // the CPU folds wind into every blade each frame and re-uploads it
    kCount
};

inline constexpr const char* kRenderPathNames[] = {"Per-blade draws", "Static batch", "Dynamic batch",
                                                   "Instanced (attributes)", "Instanced (SSBO)"};
inline constexpr const char* kRenderPathKeys[] = {"per-blade", "static-batch", "dynamic-batch",
                                                  "instanced-attrib", "instanced-ssbo"};
inline constexpr const char* kAnimationNames[] = {"Vertex shader (static data)",
                                                  "CPU per blade (re-upload)"};
inline constexpr const char* kAnimationKeys[] = {"vs", "cpu"};
inline constexpr const char* kUploadNames[] = {"glBufferSubData", "Orphaning", "Persistent ring"};
inline constexpr const char* kUploadKeys[] = {"subdata", "orphan", "persistent"};

struct GrassRenderSettings {
    RenderPath path = RenderPath::InstancedSsbo;
    Animation animation = Animation::VertexShader;
    UploadMethod upload = UploadMethod::PersistentRing;
    bool redundantState = false; // per-blade draws: rebind program/VAO and all uniforms every draw
    int cpuPathCap = 1'000'000;  // max blades for per-blade draws and batches

    bool operator==(const GrassRenderSettings&) const = default;
};

// Batches are fixed to one animation mode; the others honor the setting.
Animation effectiveAnimation(const GrassRenderSettings& settings);
// Whether the path streams CPU-written data to the GPU every frame.
bool usesStreaming(const GrassRenderSettings& settings);

class GrassRenderer {
public:
    bool init();
    void destroy();

    // CPU-side work for this frame: animation, batch building and uploads.
    // Call outside any GPU pass scope so CPU time does not appear as GPU idle time.
    void prepare(const GrassField& field, const SceneParams& scene, const GrassRenderSettings& settings,
                 CpuProfiler& cpu, GpuProfiler& gpu);
    // Issues this frame's draws. Must follow prepare() with the same arguments.
    void submit(const GrassField& field, const SceneParams& scene, const GrassRenderSettings& settings,
                CpuProfiler& cpu);

    int drawnBlades() const { return drawn_; }
    const char* error() const { return error_; }
    double fenceWaitMs() const { return fenceWaitMs_; }

private:
    struct Program {
        GLuint id = 0;
        GLint viewProj = -1, time = -1, windDir = -1, windStrength = -1, proceduralWind = -1;
        GLint sunDir = -1, cameraPos = -1, fogColor = -1, fogDensity = -1;
        GLint instanceA = -1, instanceB = -1;

        bool load(const char* vertexShader, const char* defines);
        void setScene(const SceneParams& scene, bool proceduralWind) const;
    };

    struct BatchedVertex {
        glm::vec4 posT;     // world position at rest, t
        uint32_t normal;    // 10:10:10:2 snorm
        float variation;
        float windPhase;    // per-blade phase offset
        float windWeight;   // h * t^2 for procedural wind, 0 when wind is baked in
    };
    static_assert(sizeof(BatchedVertex) == 32);

    int bladeLimit(const GrassField& field, const GrassRenderSettings& settings) const;
    void ensureBatchIndices(const GrassField& field, int blades);
    void ensureStaticBatch(const GrassField& field, int blades, CpuProfiler& cpu);
    void releaseUnused(const GrassRenderSettings& settings);
    static void writeBladeVertices(const GrassField& field, const GrassInstance& blade, glm::vec2 bend,
                                   bool proceduralWind, BatchedVertex* out);

    Program uniformProgram_, attributeProgram_, ssboProgram_, batchedProgram_;
    GLuint bladeVao_ = 0, attributeVao_ = 0, batchVao_ = 0;

    StreamBuffer instanceStream_, vertexStream_;
    std::vector<GrassInstance> animated_; // CPU-animated blades for per-blade draws

    GLuint batchIndexBuffer_ = 0, staticBatchBuffer_ = 0;
    uint64_t batchIndexBytes_ = 0, staticBatchBytes_ = 0;
    int batchIndexBlades_ = 0, staticBatchBlades_ = 0;
    uint64_t batchIndexVersion_ = ~0ull, staticBatchVersion_ = ~0ull;

    // Per-frame state handed from prepare() to submit().
    GLuint frameBuffer_ = 0;
    GLintptr frameOffset_ = 0;
    uint64_t frameBytes_ = 0;
    int drawn_ = 0;
    const char* error_ = nullptr;
    double fenceWaitMs_ = 0.0;
};
