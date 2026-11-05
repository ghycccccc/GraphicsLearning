#pragma once
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

private:
    struct Program {
        GLuint id = 0;
        GLint viewProj = -1, time = -1, windDir = -1, windStrength = -1, proceduralWind = -1;
        GLint sunDir = -1, cameraPos = -1, fogColor = -1, fogDensity = -1;
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
    static void writeBladeVertices(const GrassField& field, const GrassInstance& blade, glm::vec2 bend,
                                   bool proceduralWind, BatchedVertex* out);

    std::map<std::string, Program> grassPrograms_;
    Program batchedProgram_;
    GLuint bladeVao_ = 0, attributeVao_ = 0, batchVao_ = 0;
    InstanceLayout attributeVaoLayout_ = InstanceLayout::kCount;

    StreamBuffer staticStream_, bendStream_, vertexStream_;
    std::vector<GrassInstance> animated_; // CPU-animated blades for per-blade draws
    GrassSim sim_;

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
