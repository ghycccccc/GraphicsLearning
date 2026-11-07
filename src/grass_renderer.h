#pragma once
#include "culling.h"
#include "gpu_culler.h"
#include "grass.h"
#include "grass_sim.h"
#include "stream_buffer.h"

#include <cstdint>
#include <map>
#include <string>
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
    VertexShader,      // B3: stateless wind computed in the vertex shader, no per-frame data
    CpuWind,           // B1: stateless wind computed on the CPU, re-uploaded every frame
    SpringCpu,         // B1: stateful spring sim on one CPU thread, re-uploaded every frame
    SpringCpuParallel, // B1: same, spread over all CPU cores
    SpringCompute,     // B2: same sim in a compute shader, state never leaves the GPU
    kCount
};

// What the CPU paths re-upload each frame.
enum class UploadScope {
    ChangedOnly, // just the bends (8 B or 4 B per blade)
    Full,        // static data + bends, as an engine that re-sends whole instance records would
    kCount
};

enum class Trample { Off, Circle, FollowCamera, kCount };

// Who decides which blades to draw (instanced SSBO path only).
enum class CullMode {
    None,              // draw every blade
    CpuBlades,         // C1: CPU tests every blade, uploads the visible id list
    CpuBladesParallel, // C1: same on all cores (two-pass compaction)
    CpuChunks,         // C1: CPU tests 8 m chunks, emits one indirect command per visible chunk
    Gpu,               // C2/C3: compute shader culls (+ LOD), writes its own indirect commands
    kCount
};

inline constexpr const char* kRenderPathNames[] = {"Per-blade draws", "Static batch", "Dynamic batch",
                                                   "Instanced (attributes)", "Instanced (SSBO)"};
inline constexpr const char* kRenderPathKeys[] = {"per-blade", "static-batch", "dynamic-batch",
                                                  "instanced-attrib", "instanced-ssbo"};
inline constexpr const char* kAnimationNames[] = {"Vertex shader wind (stateless)", "CPU wind, stateless",
                                                  "Spring sim, CPU 1 thread", "Spring sim, CPU all cores",
                                                  "Spring sim, compute shader"};
inline constexpr const char* kAnimationKeys[] = {"vs", "cpu", "spring-cpu", "spring-mt", "spring-gpu"};
inline constexpr const char* kUploadNames[] = {"glBufferSubData", "Orphaning", "Persistent ring"};
inline constexpr const char* kUploadKeys[] = {"subdata", "orphan", "persistent"};
inline constexpr const char* kUploadScopeNames[] = {"Changed data only (bends)", "Full instance data"};
inline constexpr const char* kUploadScopeKeys[] = {"changed", "full"};
inline constexpr const char* kTrampleNames[] = {"Off", "Scripted circle", "Follow camera"};
inline constexpr const char* kTrampleKeys[] = {"off", "circle", "camera"};
inline constexpr const char* kCullNames[] = {"None", "CPU per blade, 1 thread", "CPU per blade, all cores",
                                             "CPU chunks + multi-draw indirect", "GPU compute + indirect"};
inline constexpr const char* kCullKeys[] = {"none", "cpu", "cpu-mt", "cpu-chunks", "gpu"};

struct GrassRenderSettings {
    RenderPath path = RenderPath::InstancedSsbo;
    Animation animation = Animation::VertexShader;
    UploadMethod upload = UploadMethod::PersistentRing;
    UploadScope uploadScope = UploadScope::ChangedOnly;
    InstanceLayout layout = InstanceLayout::Float32;
    Trample trample = Trample::Circle;
    Readback readback = Readback::None;
    float stiffness = 40.0f;
    float damping = 4.0f;
    CullMode cull = CullMode::None;
    GpuCullSettings gpuCull;     // compaction, LOD, readback anti-pattern
    float maxDistance = 0.0f;    // 0 = no distance culling
    bool freezeCulling = false;  // keep culling with the camera of the moment it was frozen
    bool redundantState = false; // per-blade draws: rebind program/VAO and all uniforms every draw
    int cpuPathCap = 1'000'000;  // max blades for per-blade draws and batches

    bool operator==(const GrassRenderSettings&) const = default;
};

// The animation actually used: batches are fixed, per-blade draws only know stateless wind.
Animation effectiveAnimation(const GrassRenderSettings& settings);
// Whether the path streams CPU-written data to the GPU every frame.
bool usesStreaming(const GrassRenderSettings& settings);
bool isInstanced(RenderPath path);
bool isSpring(Animation animation);
// Culling needs the visible-id indirection, which only the SSBO path has.
CullMode effectiveCull(const GrassRenderSettings& settings);

// What the last frame's culling decided. GPU numbers arrive asynchronously (ageFrames late).
struct CullStats {
    uint32_t visible[GrassField::kLodCount] = {};
    uint32_t visibleTotal = 0;
    int ageFrames = -1;     // -1: unknown
    uint32_t chunksDrawn = 0;
    int chunkCount = 0;
};

class GrassRenderer {
public:
    bool init();
    void destroy();

    // CPU-side work for this frame: animation, simulation, batch building and uploads.
    // Call outside any draw pass scope so CPU time does not appear as GPU idle time.
    void prepare(const GrassField& field, const SceneParams& scene, const GrassRenderSettings& settings,
                 CpuProfiler& cpu, GpuProfiler& gpu);
    // Issues this frame's draws. Must follow prepare() with the same arguments.
    void submit(const GrassField& field, const SceneParams& scene, const GrassRenderSettings& settings,
                CpuProfiler& cpu);

    int drawnBlades() const { return drawn_; }
    const char* error() const { return error_; }
    const GrassSim& sim() const { return sim_; }
    const CullStats& cullStats() const { return cullStats_; }
    // The camera culling used this frame (differs from the render camera while frozen).
    const CullView& cullView() const { return cullView_; }
    // Sets the culling camera directly; it stays while settings.freezeCulling is on.
    void setCullView(const glm::mat4& viewProj, const glm::vec3& cameraPos) {
        cullView_.viewProj = viewProj;
        cullView_.cameraPos = cameraPos;
        hasCullView_ = true;
    }
    double cullReadbackStallMs() const { return gpuCuller_.readbackStallMs(); }

private:
    struct Program {
        GLuint id = 0;
        GLint viewProj = -1, time = -1, windDir = -1, windStrength = -1, proceduralWind = -1;
        GLint sunDir = -1, cameraPos = -1, fogColor = -1, fogDensity = -1, colorByDraw = -1;
        GLint instanceA = -1, instanceB = -1;

        bool load(const char* vertexShader, const std::string& defines);
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

    // grass.vert variants are compiled on first use and cached by their #define block.
    const Program& grassProgram(const std::string& defines);
    void configureAttributeVao(InstanceLayout layout);
    int bladeLimit(const GrassField& field, const GrassRenderSettings& settings) const;
    void ensureBatchIndices(const GrassField& field, int blades);
    void ensureStaticBatch(const GrassField& field, int blades, CpuProfiler& cpu);
    void releaseUnused(const GrassRenderSettings& settings);
    void prepareInstanced(const GrassField& field, const SceneParams& scene, const GrassRenderSettings& settings,
                          CpuProfiler& cpu, GpuProfiler& gpu);
    void prepareCulling(const GrassField& field, const SceneParams& scene, const GrassRenderSettings& settings,
                        CpuProfiler& cpu, GpuProfiler& gpu);
    void drawCulled(const GrassField& field, const GrassRenderSettings& settings);
    static void writeBladeVertices(const GrassField& field, const GrassInstance& blade, glm::vec2 bend,
                                   bool proceduralWind, BatchedVertex* out);

    std::map<std::string, Program> grassPrograms_;
    Program batchedProgram_;
    GLuint bladeVao_ = 0, attributeVao_ = 0, batchVao_ = 0;
    InstanceLayout attributeVaoLayout_ = InstanceLayout::kCount;

    StreamBuffer staticStream_, bendStream_, vertexStream_;
    std::vector<GrassInstance> animated_; // CPU-animated blades for per-blade draws
    GrassSim sim_;

    // Culling
    StreamBuffer visibleStream_, commandStream_; // CPU-written id lists / indirect commands
    ChunkGrid chunks_;
    GpuCuller gpuCuller_;
    CullView cullView_;
    bool hasCullView_ = false;
    CullStats cullStats_;
    GLuint frameVisibleBuffer_ = 0;
    GLintptr frameVisibleOffset_ = 0, frameCommandOffset_ = 0;
    uint32_t frameVisibleCount_ = 0, frameCommandCount_ = 0;

    GLuint batchIndexBuffer_ = 0, staticBatchBuffer_ = 0;
    uint64_t batchIndexBytes_ = 0, staticBatchBytes_ = 0;
    int batchIndexBlades_ = 0, staticBatchBlades_ = 0;
    uint64_t batchIndexVersion_ = ~0ull, staticBatchVersion_ = ~0ull;

    // Per-frame state handed from prepare() to submit().
    GLuint frameStaticBuffer_ = 0, frameBendBuffer_ = 0, frameVertexBuffer_ = 0;
    GLintptr frameStaticOffset_ = 0, frameBendOffset_ = 0, frameVertexOffset_ = 0;
    bool frameDynamicBend_ = false;
    int drawn_ = 0;
    const char* error_ = nullptr;
};
