#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

#include "camera.h"
#include "counters.h"
#include "gl_util.h"
#include "grass.h"
#include "grass_renderer.h"
#include "profiler.h"
#include "render_target.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

// Ask Optimus / PowerXpress laptops to run this process on the discrete GPU.
// The InstancingLab_iGPU target omits this to benchmark the integrated GPU.
#ifndef LAB_INTEGRATED_GPU
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

namespace {

using Clock = std::chrono::steady_clock;

double msBetween(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

constexpr int kMaxInstances = 16'000'000;
// The app keeps at most this many frames queued on the GPU. Waiting for the oldest one
// happens at a known point (the frame fence), so time spent there measures GPU-boundness.
constexpr int kMaxFramesInFlight = 2;
constexpr const char* kBudgetLabels[] = {"None", "30 fps (33.3 ms)", "60 fps (16.7 ms)",
                                         "120 fps (8.3 ms)", "144 fps (6.9 ms)"};
constexpr float kBudgetMs[] = {0.0f, 1000.0f / 30.0f, 1000.0f / 60.0f, 1000.0f / 120.0f,
                               1000.0f / 144.0f};

struct AppState {
    int instanceCount = 100000;
    float density = 40.0f; // blades per m^2
    uint32_t seed = 1;
    bool regenerate = true;
    bool regenerateFailed = false;
    int segments = 4;
    float renderScale = 1.0f; // per axis
    GrassRenderSettings render;
    int budgetIndex = 2;
    bool vsync = false;
    float windStrength = 0.35f;
    float windAngle = 0.6f;
    float fogDensity = 0.012f;
    int statsWindow = 240;
    bool resetStats = false;
    int warmupFrames = 0; // frames left to skip after a reset
    float gameLogicMs = 0.0f; // simulated per-frame CPU work (gameplay, AI, physics...)
    bool hasCameraStart = false;
    float cameraStart[5] = {}; // x, y, z, yaw, pitch

    // Benchmark mode: measure, append one CSV row, exit.
    int benchFrames = 0;
    float benchSeconds = 20.0f;
    std::string csvPath = "bench/results.csv";
};

// Everything the UI reads, captured at the end of the previous frame.
struct FrameReport {
    FrameCounters counters;
    double cpuWorkMs = 0.0;
    double gpuWaitMs = 0.0;
    double swapMs = 0.0;
    uint64_t vramTotalKb = 0;
    uint64_t vramAvailableKb = 0;
    int internalWidth = 0;
    int internalHeight = 0;
};

template <typename E, size_t N>
bool parseEnum(const char* value, const char* const (&keys)[N], E& out) {
    for (size_t i = 0; i < N; ++i) {
        if (std::strcmp(value, keys[i]) == 0) {
            out = E(i);
            return true;
        }
    }
    std::fprintf(stderr, "Unknown value %s\n", value);
    return false;
}

// Command-line overrides, e.g. --instances 4000000 --path instanced-ssbo --bench-frames 120
void parseArgs(int argc, char** argv, AppState& app) {
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        const char* value = argv[i + 1];
        if (key == "--instances") app.instanceCount = std::clamp(std::atoi(value), 1, kMaxInstances);
        else if (key == "--density") app.density = float(std::atof(value));
        else if (key == "--segments") app.segments = std::clamp(std::atoi(value), 1, GrassField::kMaxSegments);
        else if (key == "--scale") app.renderScale = std::clamp(float(std::atof(value)), 0.25f, 4.0f);
        else if (key == "--path") parseEnum(value, kRenderPathKeys, app.render.path);
        else if (key == "--animation") parseEnum(value, kAnimationKeys, app.render.animation);
        else if (key == "--upload") parseEnum(value, kUploadKeys, app.render.upload);
        else if (key == "--upload-scope") parseEnum(value, kUploadScopeKeys, app.render.uploadScope);
        else if (key == "--layout") parseEnum(value, kLayoutKeys, app.render.layout);
        else if (key == "--trample") parseEnum(value, kTrampleKeys, app.render.trample);
        else if (key == "--readback") parseEnum(value, kReadbackKeys, app.render.readback);
        else if (key == "--stiffness") app.render.stiffness = float(std::atof(value));
        else if (key == "--damping") app.render.damping = float(std::atof(value));
        else if (key == "--redundant") app.render.redundantState = std::atoi(value) != 0;
        else if (key == "--game-logic-ms") app.gameLogicMs = std::clamp(float(std::atof(value)), 0.0f, 100.0f);
        else if (key == "--camera") {
            // x,y,z,yaw,pitch (radians)
            if (std::sscanf(value, "%f,%f,%f,%f,%f", &app.cameraStart[0], &app.cameraStart[1], &app.cameraStart[2],
                            &app.cameraStart[3], &app.cameraStart[4]) != 5) {
                std::fprintf(stderr, "--camera expects x,y,z,yaw,pitch\n");
            } else {
                app.hasCameraStart = true;
            }
        }
        else if (key == "--cap") app.render.cpuPathCap = std::clamp(std::atoi(value), 1, kMaxInstances);
        else if (key == "--bench-frames") app.benchFrames = std::clamp(std::atoi(value), 5, FrameHistory::kCapacity);
        else if (key == "--bench-seconds") app.benchSeconds = float(std::atof(value));
        else if (key == "--csv") app.csvPath = value;
        else std::fprintf(stderr, "Unknown argument %s\n", key.c_str());
    }
}

std::string withSeparators(uint64_t value) {
    std::string digits = std::to_string(value);
    for (int i = int(digits.size()) - 3; i > 0; i -= 3) digits.insert(size_t(i), ",");
    return digits;
}

// The CPU waiting on the GPU (frame fence or a blocking swap) means the GPU is the limiter;
// a CPU busy for the whole frame without waiting means the CPU is.
const char* verdictFor(bool vsync, float frameMs, float cpuMs, float waitMs) {
    if (frameMs <= 0.0f) return "-";
    if (vsync && std::fabs(frameMs - 1000.0f / 60.0f) < 1.0f) return "VSync-limited";
    if (waitMs / frameMs > 0.15f) return "GPU-bound";
    if (cpuMs / frameMs > 0.85f) return "CPU-bound";
    return "Neither saturated";
}

void statsRow(const char* label, const FrameHistory::Stats& s) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(label);
    for (float v : {s.avg, s.p50, s.p95, s.p99, s.max}) {
        ImGui::TableNextColumn();
        ImGui::Text("%.2f", v);
    }
}

template <typename Results>
double averageMs(const Results& results, const char* name) {
    for (const auto& r : results) {
        if (std::strcmp(r.name, name) == 0) return r.avgMs;
    }
    return 0.0;
}

void appendCsvRow(const AppState& app, const GrassField& grass, const GrassRenderer& renderer,
                  const FrameHistory& history, const FrameReport& report, const CpuProfiler& cpuProfiler,
                  const GpuProfiler& gpuProfiler, const char* gpuName) {
    const int window = history.size();
    const auto frame = history.stats(FrameHistory::Frame, window);
    const auto cpu = history.stats(FrameHistory::Cpu, window);
    const auto gpu = history.stats(FrameHistory::Gpu, window);
    const auto wait = history.stats(FrameHistory::Wait, window);

    static const char* kHeader =
        "gpu,path,animation,upload,upload_scope,layout,trample,readback,redundant,instances,drawn,segments,"
        "scale,frames,frame_avg,frame_p50,frame_p95,frame_p99,cpu_avg,gpu_avg,wait_avg,sim_cpu_ms,sim_gpu_ms,"
        "readback_ms,draw_calls,triangles,upload_mb,verdict,game_logic_ms";
    const std::filesystem::path path(app.csvPath);
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    const bool exists = std::filesystem::exists(path);
    if (exists) {
        std::ifstream in(path);
        std::string header;
        std::getline(in, header);
        if (header != kHeader) {
            std::fprintf(stderr, "%s has a different column layout; use a new --csv file\n", app.csvPath.c_str());
            return;
        }
    }
    std::ofstream out(path, std::ios::app);
    if (!exists) out << kHeader << '\n';

    const GrassRenderSettings& r = app.render;
    const Animation animation = effectiveAnimation(r);
    const bool instanced = isInstanced(r.path);
    const double simCpuMs = averageMs(cpuProfiler.results(), "Simulate (CPU)") +
                            averageMs(cpuProfiler.results(), "CPU wind") +
                            averageMs(cpuProfiler.results(), "CPU build vertices");
    out << '"' << gpuName << "\"," << kRenderPathKeys[int(r.path)] << ',' << kAnimationKeys[int(animation)] << ','
        << (usesStreaming(r) ? kUploadKeys[int(r.upload)] : "none") << ','
        << (usesStreaming(r) && instanced ? kUploadScopeKeys[int(r.uploadScope)] : "none") << ','
        << (instanced ? kLayoutKeys[int(r.layout)] : "none") << ','
        << (instanced && isSpring(animation) ? kTrampleKeys[int(r.trample)] : "none") << ','
        << (instanced && animation == Animation::SpringCompute ? kReadbackKeys[int(r.readback)] : "none") << ','
        << int(r.redundantState) << ',' << grass.count() << ',' << renderer.drawnBlades() << ','
        << grass.segments() << ',' << app.renderScale << ',' << window << ',' << frame.avg << ',' << frame.p50 << ','
        << frame.p95 << ',' << frame.p99 << ',' << cpu.avg << ',' << gpu.avg << ',' << wait.avg << ',' << simCpuMs
        << ',' << averageMs(gpuProfiler.results(), "Simulate") << ',' << renderer.sim().readbackStallMs() << ','
        << report.counters.drawCalls << ',' << report.counters.triangles << ','
        << double(report.counters.bytesUploaded) / (1024.0 * 1024.0) << ','
        << verdictFor(false, frame.avg, cpu.avg, wait.avg) << ',' << app.gameLogicMs << '\n';
    std::printf("bench: %s %d blades: frame %.2f ms (cpu %.2f, gpu %.2f, wait %.2f)\n",
                kRenderPathKeys[int(r.path)], grass.count(), frame.avg, cpu.avg, gpu.avg, wait.avg);
}

void drawUi(AppState& app, FlyCamera& camera, const GrassField& grass, const GrassRenderer& grassRenderer,
            const GpuProfiler& gpu, const CpuProfiler& cpu, const FrameHistory& history,
            const FrameReport& report, const char* gpuName) {
    ImGui::SetNextWindowPos({10, 10}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({460, 880}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Instancing Lab");

    ImGui::TextDisabled("%s", gpuName);
#ifdef LAB_INTEGRATED_GPU
    ImGui::TextColored({0.5f, 0.8f, 1.0f, 1.0f}, "iGPU build (no discrete GPU hint)");
#else
    if (!std::strstr(gpuName, "NVIDIA") && !std::strstr(gpuName, "AMD") &&
        !std::strstr(gpuName, "Radeon")) {
        ImGui::TextColored({1.0f, 0.45f, 0.3f, 1.0f}, "Warning: not running on the discrete GPU");
    }
#endif

    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
        GrassRenderSettings& r = app.render;
        int path = int(r.path);
        if (ImGui::Combo("Path", &path, kRenderPathNames, IM_ARRAYSIZE(kRenderPathNames))) r.path = RenderPath(path);

        const bool batch = r.path == RenderPath::StaticBatch || r.path == RenderPath::DynamicBatch;
        const bool instanced = isInstanced(r.path);
        if (batch) {
            ImGui::TextDisabled("Animation: %s (fixed for this path)", kAnimationNames[int(effectiveAnimation(r))]);
        } else {
            // Per-blade draws only offer the stateless modes; the spring sim is instanced-only.
            int animation = int(effectiveAnimation(r));
            const int choices = instanced ? IM_ARRAYSIZE(kAnimationNames) : 2;
            if (ImGui::Combo("Animation", &animation, kAnimationNames, choices)) r.animation = Animation(animation);
        }
        const Animation animation = effectiveAnimation(r);
        if (instanced) {
            int layout = int(r.layout);
            if (ImGui::Combo("Instance layout", &layout, kLayoutNames, IM_ARRAYSIZE(kLayoutNames)))
                r.layout = InstanceLayout(layout);
        }
        if (usesStreaming(r)) {
            int upload = int(r.upload);
            if (ImGui::Combo("Upload", &upload, kUploadNames, IM_ARRAYSIZE(kUploadNames)))
                r.upload = UploadMethod(upload);
            if (instanced) {
                int scope = int(r.uploadScope);
                if (ImGui::Combo("Upload scope", &scope, kUploadScopeNames, IM_ARRAYSIZE(kUploadScopeNames)))
                    r.uploadScope = UploadScope(scope);
            }
        }
        if (instanced && isSpring(animation)) {
            ImGui::SliderFloat("Stiffness", &r.stiffness, 5.0f, 200.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Damping", &r.damping, 0.0f, 30.0f, "%.1f");
            int trample = int(r.trample);
            if (ImGui::Combo("Trample", &trample, kTrampleNames, IM_ARRAYSIZE(kTrampleNames)))
                r.trample = Trample(trample);
            if (animation == Animation::SpringCompute) {
                int readback = int(r.readback);
                if (ImGui::Combo("CPU readback", &readback, kReadbackNames, IM_ARRAYSIZE(kReadbackNames)))
                    r.readback = Readback(readback);
            }
            const GrassSim& sim = grassRenderer.sim();
            if (sim.flattenedAgeFrames() >= 0) {
                ImGui::Text("Flattened blades: %s (%d frame%s old)", withSeparators(uint64_t(sim.flattenedBlades())).c_str(),
                            sim.flattenedAgeFrames(), sim.flattenedAgeFrames() == 1 ? "" : "s");
            } else {
                ImGui::TextDisabled("Flattened blades: unknown to the CPU (no readback)");
            }
            if (animation == Animation::SpringCompute && r.readback == Readback::Sync) {
                ImGui::Text("Readback stall: %.2f ms", sim.readbackStallMs());
            }
            if (animation == Animation::SpringCpuParallel) {
                ImGui::TextDisabled("%u hardware threads", std::thread::hardware_concurrency());
            }
        }
        if (r.path == RenderPath::PerBladeDraws) {
            ImGui::Checkbox("Redundant state per draw", &r.redundantState);
        }
        if (r.path == RenderPath::PerBladeDraws || batch) {
            ImGui::SliderInt("CPU path cap", &r.cpuPathCap, 1000, kMaxInstances, "%d", ImGuiSliderFlags_Logarithmic);
        }
        if (grassRenderer.drawnBlades() < grass.count()) {
            ImGui::TextColored({1.0f, 0.75f, 0.3f, 1.0f}, "Drawing %s of %s blades (cap / memory budget)",
                               withSeparators(uint64_t(grassRenderer.drawnBlades())).c_str(),
                               withSeparators(uint64_t(grass.count())).c_str());
        }
        if (grassRenderer.error()) ImGui::TextColored({1.0f, 0.45f, 0.3f, 1.0f}, "%s", grassRenderer.error());
    }

    if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderInt("Instances", &app.instanceCount, 1000, kMaxInstances, "%d",
                         ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemDeactivatedAfterEdit()) app.regenerate = true;
        ImGui::SliderFloat("Density /m^2", &app.density, 5.0f, 400.0f, "%.0f",
                           ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemDeactivatedAfterEdit()) app.regenerate = true;
        if (app.regenerateFailed) {
            ImGui::TextColored({1.0f, 0.45f, 0.3f, 1.0f}, "Out of GPU memory, field is empty");
        }
        ImGui::Text("Field %.0f x %.0f m   blade: %d verts, %d tris", grass.fieldSize(),
                    grass.fieldSize(), grass.verticesPerBlade(), grass.trianglesPerBlade());
        ImGui::SliderFloat("Wind strength", &app.windStrength, 0.0f, 1.5f);
        ImGui::SliderAngle("Wind direction", &app.windAngle, -180.0f, 180.0f);
        ImGui::SliderFloat("Fog density", &app.fogDensity, 0.0f, 0.05f, "%.3f");
        ImGui::SliderFloat("Camera speed", &camera.speed, 1.0f, 100.0f, "%.0f",
                           ImGuiSliderFlags_Logarithmic);
        if (ImGui::Checkbox("VSync", &app.vsync)) {
            glfwSwapInterval(app.vsync ? 1 : 0);
            app.resetStats = true;
        }
        ImGui::TextDisabled("RMB drag: look   WASD/QE: move   Shift: fast");
    }

    if (ImGui::CollapsingHeader("Stress")) {
        ImGui::SliderInt("Blade segments", &app.segments, 1, GrassField::kMaxSegments);
        ImGui::SliderFloat("Render scale", &app.renderScale, 0.25f, 4.0f, "%.2fx");
        const double mpix = double(report.internalWidth) * report.internalHeight * 1e-6;
        ImGui::Text("Internal %d x %d (%.1f MPix)", report.internalWidth, report.internalHeight, mpix);
        ImGui::Text("Triangles in scene %s",
                    withSeparators(uint64_t(grass.count()) * uint64_t(grass.trianglesPerBlade())).c_str());
    }

    if (ImGui::CollapsingHeader("Frame timing", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderInt("Stats window", &app.statsWindow, 30, FrameHistory::kCapacity);
        const auto frame = history.stats(FrameHistory::Frame, app.statsWindow);
        const auto cpuStats = history.stats(FrameHistory::Cpu, app.statsWindow);
        const auto gpuStats = history.stats(FrameHistory::Gpu, app.statsWindow);
        const auto waitStats = history.stats(FrameHistory::Wait, app.statsWindow);

        if (ImGui::BeginTable("timing", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            for (const char* h : {"ms", "avg", "p50", "p95", "p99", "max"})
                ImGui::TableSetupColumn(h);
            ImGui::TableHeadersRow();
            statsRow("Frame", frame);
            statsRow("CPU work", cpuStats);
            statsRow("GPU work", gpuStats);
            statsRow("Wait on GPU", waitStats);
            ImGui::EndTable();
        }
        ImGui::Text("avg %.0f fps   1%% low %.0f fps", 1000.0f / frame.avg, 1000.0f / frame.worst1PctAvg);

        ImGui::Combo("Frame budget", &app.budgetIndex, kBudgetLabels, IM_ARRAYSIZE(kBudgetLabels));
        const float budgetMs = kBudgetMs[app.budgetIndex];
        if (budgetMs > 0.0f) {
            const float headroom = budgetMs - frame.avg;
            const float missed = history.fractionAbove(FrameHistory::Frame, app.statsWindow, budgetMs);
            const ImVec4 color = headroom >= 0.0f && missed < 0.01f ? ImVec4(0.5f, 0.9f, 0.5f, 1.0f)
                                                                    : ImVec4(1.0f, 0.45f, 0.3f, 1.0f);
            ImGui::TextColored(color, "Headroom %.2f ms (%.0f%% of budget)   over budget: %.1f%% of frames",
                               headroom, 100.0f * headroom / budgetMs, 100.0f * missed);
        }

        ImGui::SliderFloat("Game logic (CPU)", &app.gameLogicMs, 0.0f, 30.0f, "%.1f ms");
        if (ImGui::IsItemDeactivatedAfterEdit()) app.resetStats = true;
        ImGui::TextDisabled("Busy-waits on the main thread after the grass sim/readback,\n"
                            "standing in for gameplay, AI and physics.");

        ImGui::Text("Verdict: %s", verdictFor(app.vsync, frame.avg, cpuStats.avg, waitStats.avg));
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset stats")) app.resetStats = true;
        const float cpuLoad = frame.avg > 0.0f ? cpuStats.avg / frame.avg : 0.0f;
        const float gpuLoad = frame.avg > 0.0f ? gpuStats.avg / frame.avg : 0.0f;
        const float waitLoad = frame.avg > 0.0f ? waitStats.avg / frame.avg : 0.0f;
        ImGui::ProgressBar(cpuLoad, {-1, 0}, ("CPU busy " + std::to_string(int(cpuLoad * 100)) + "%").c_str());
        ImGui::ProgressBar(gpuLoad, {-1, 0}, ("GPU busy " + std::to_string(int(gpuLoad * 100)) + "%").c_str());
        ImGui::ProgressBar(waitLoad, {-1, 0}, ("CPU waiting on GPU " + std::to_string(int(waitLoad * 100)) + "%").c_str());
        ImGui::TextDisabled("GPU busy is inflated when the GPU sits idle inside a pass\n"
                            "waiting for draws (e.g. per-blade draws); trust the wait bar.");

        if (ImPlot::BeginPlot("##frametimes", {-1, 190})) {
            ImPlot::SetupAxes(nullptr, "ms", ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, FrameHistory::kCapacity, ImGuiCond_Always);
            ImPlot::SetupLegend(ImPlotLocation_NorthWest);
            const int n = history.size(), off = history.offset();
            ImPlot::PlotLine("Frame", history.data(FrameHistory::Frame), n, 1.0, 0.0, 0, off);
            ImPlot::PlotLine("CPU", history.data(FrameHistory::Cpu), n, 1.0, 0.0, 0, off);
            ImPlot::PlotLine("GPU", history.data(FrameHistory::Gpu), n, 1.0, 0.0, 0, off);
            ImPlot::PlotLine("Wait", history.data(FrameHistory::Wait), n, 1.0, 0.0, 0, off);
            if (budgetMs > 0.0f) {
                const double budgetLine = budgetMs;
                ImPlot::PlotInfLines("Budget", &budgetLine, 1, ImPlotInfLinesFlags_Horizontal);
            }
            ImPlot::EndPlot();
        }
    }

    if (ImGui::CollapsingHeader("GPU passes", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::BeginTable("gpu", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("pass");
            ImGui::TableSetupColumn("ms");
            ImGui::TableSetupColumn("avg ms");
            ImGui::TableHeadersRow();
            for (const auto& r : gpu.results()) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Indent(float(r.depth) * 12.0f + 0.001f);
                ImGui::TextUnformatted(r.name);
                ImGui::Unindent(float(r.depth) * 12.0f + 0.001f);
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", r.ms);
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", r.avgMs);
            }
            ImGui::EndTable();
        }
        ImGui::TextDisabled("Timestamp queries, %d frames latency, %d late readbacks skipped",
                            GpuProfiler::kFrames - 1, gpu.missedReadbacks());
    }

    if (ImGui::CollapsingHeader("CPU stages", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::BeginTable("cpu", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("stage");
            ImGui::TableSetupColumn("ms");
            ImGui::TableSetupColumn("avg ms");
            ImGui::TableHeadersRow();
            for (const auto& r : cpu.results()) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(r.name);
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", r.ms);
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", r.avgMs);
            }
            ImGui::EndTable();
        }
        ImGui::TextDisabled("Submit = time inside GL calls (driver overhead), not GPU time");
    }

    if (ImGui::CollapsingHeader("Counters", ImGuiTreeNodeFlags_DefaultOpen)) {
        const FrameCounters& c = report.counters;
        ImGui::Text("Draw calls        %s", withSeparators(c.drawCalls).c_str());
        ImGui::Text("State changes     %s", withSeparators(c.stateChanges).c_str());
        ImGui::Text("Uniform updates   %s", withSeparators(c.uniformUpdates).c_str());
        ImGui::Text("Instances         %s", withSeparators(c.instances).c_str());
        ImGui::Text("Triangles         %s", withSeparators(c.triangles).c_str());
        ImGui::Text("Uploaded / frame  %.2f MB", double(c.bytesUploaded) / (1024.0 * 1024.0));
        ImGui::Text("App GPU buffers   %.2f MB", double(g_residentBytes) / (1024.0 * 1024.0));
        if (report.vramTotalKb) {
            ImGui::Text("VRAM in use       %.0f / %.0f MB (all processes)",
                        double(report.vramTotalKb - report.vramAvailableKb) / 1024.0,
                        double(report.vramTotalKb) / 1024.0);
        }
    }

    ImGui::End();
}

} // namespace

int main(int argc, char** argv) {
    AppState app;
    parseArgs(argc, argv, app);
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifndef NDEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif
    GLFWwindow* window = glfwCreateWindow(1600, 900, "Instancing Lab", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Failed to create an OpenGL 4.6 core context\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::fprintf(stderr, "glewInit failed\n");
        return 1;
    }
    glGetError(); // GLEW can leave a spurious GL_INVALID_ENUM on core contexts
    installDebugOutput();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");

    GrassField grass;
    GrassRenderer grassRenderer;
    Ground ground;
    if (!grass.init() || !grassRenderer.init() || !ground.init()) return 1;

    GpuProfiler gpu;
    gpu.init();
    CpuProfiler cpu;
    FrameHistory history;
    FlyCamera camera;
    if (app.hasCameraStart) {
        camera.position = {app.cameraStart[0], app.cameraStart[1], app.cameraStart[2]};
        camera.yaw = app.cameraStart[3];
        camera.pitch = app.cameraStart[4];
    }
    FrameReport report;
    RenderTarget sceneTarget;
    GLsync frameFences[kMaxFramesInFlight] = {};
    GrassRenderSettings lastRenderSettings = app.render;
    if (app.benchFrames > 0) app.statsWindow = app.benchFrames;

    const char* gpuName = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    std::printf("Renderer: %s\nVersion:  %s\n", gpuName,
                reinterpret_cast<const char*>(glGetString(GL_VERSION)));
    const bool hasNvMemInfo = GLEW_NVX_gpu_memory_info;

    const SceneParams defaults;
    const Clock::time_point startTime = Clock::now();
    Clock::time_point prevFrameStart = startTime;
    Clock::time_point statsResetTime = startTime;
    uint64_t frameIndex = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        int fbWidth = 0, fbHeight = 0;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        if (fbWidth == 0 || fbHeight == 0) { // minimized
            glfwWaitEvents();
            prevFrameStart = Clock::now();
            continue;
        }

        // Frame interval is measured start-to-start, so it includes all waiting.
        const Clock::time_point frameStart = Clock::now();
        const double frameMs = msBetween(prevFrameStart, frameStart);
        prevFrameStart = frameStart;
        if (app.warmupFrames > 0) {
            --app.warmupFrames;
        } else if (frameIndex > 0) {
            history.push(float(frameMs), float(report.cpuWorkMs), float(gpu.frameMs()),
                         float(report.gpuWaitMs + report.swapMs));
        }

        // Benchmark mode: enough samples of the current configuration -> record and quit.
        if (app.benchFrames > 0 && app.warmupFrames == 0 && history.size() >= 5 &&
            (history.size() >= app.benchFrames ||
             msBetween(statsResetTime, frameStart) > app.benchSeconds * 1000.0)) {
            appendCsvRow(app, grass, grassRenderer, history, report, cpu, gpu, gpuName);
            break;
        }

        // Wait until the frame that last used this slot has finished on the GPU.
        GLsync& frameFence = frameFences[frameIndex % kMaxFramesInFlight];
        if (frameFence) {
            while (glClientWaitSync(frameFence, GL_SYNC_FLUSH_COMMANDS_BIT, 1'000'000'000) == GL_TIMEOUT_EXPIRED) {
            }
            glDeleteSync(frameFence);
            frameFence = nullptr;
        }
        const Clock::time_point workStart = Clock::now();
        report.gpuWaitMs = msBetween(frameStart, workStart);

        report.counters = g_counters;
        g_counters = {};
        gpu.beginFrame();
        cpu.beginFrame();
        if (hasNvMemInfo && frameIndex % 30 == 0) {
            GLint total = 0, available = 0;
            glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX, &total);
            glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &available);
            report.vramTotalKb = uint64_t(total);
            report.vramAvailableKb = uint64_t(available);
        }

        {
            CpuScope scope(cpu, "UI build");
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            drawUi(app, camera, grass, grassRenderer, gpu, cpu, history, report, gpuName);
        }
        const int internalWidth = std::max(1, int(std::lround(fbWidth * app.renderScale)));
        const int internalHeight = std::max(1, int(std::lround(fbHeight * app.renderScale)));
        if (internalWidth != sceneTarget.width() || internalHeight != sceneTarget.height()) {
            app.resetStats = true;
        }
        if (app.regenerate) {
            CpuScope scope(cpu, "Regenerate field");
            app.regenerateFailed = !grass.regenerate(app.instanceCount, app.density, app.seed);
            app.regenerate = false;
            app.resetStats = true;
            if (app.regenerateFailed && app.benchFrames > 0) break;
        }
        if (app.segments != grass.segments()) {
            grass.setSegments(app.segments);
            app.resetStats = true;
        }
        if (app.render.layout != grass.layout()) {
            CpuScope scope(cpu, "Re-encode blades");
            app.regenerateFailed = !grass.setLayout(app.render.layout);
            app.resetStats = true;
        }
        if (!(app.render == lastRenderSettings)) {
            lastRenderSettings = app.render;
            app.resetStats = true;
        }
        if (app.resetStats) {
            // Drop samples from the old configuration, including in-flight GPU timings.
            history.clear();
            app.warmupFrames = GpuProfiler::kFrames + 1;
            app.resetStats = false;
            statsResetTime = frameStart;
        }

        const ImGuiIO& io = ImGui::GetIO();
        camera.update(window, float(std::fmin(frameMs * 0.001, 0.1)), io.WantCaptureMouse,
                      io.WantCaptureKeyboard);

        sceneTarget.resize(internalWidth, internalHeight);
        report.internalWidth = internalWidth;
        report.internalHeight = internalHeight;

        SceneParams scene;
        const float aspect = float(fbWidth) / float(fbHeight);
        scene.viewProj = camera.projection(aspect) * camera.view();
        scene.cameraPos = camera.position;
        scene.time = float(msBetween(startTime, frameStart) * 0.001);
        scene.windDir = {std::cos(app.windAngle), std::sin(app.windAngle)};
        scene.windStrength = app.windStrength;
        scene.sunDir = glm::normalize(glm::vec3(0.4f, 0.8f, 0.3f));
        scene.fogColor = defaults.fogColor;
        scene.fogDensity = app.fogDensity;
        if (app.render.trample == Trample::Circle) {
            const float radius = std::fmin(grass.fieldSize() * 0.3f, 12.0f);
            scene.trampleCenter = {radius * std::cos(scene.time * 0.5f), radius * std::sin(scene.time * 0.5f)};
            scene.trampleRadius = 1.5f;
        } else if (app.render.trample == Trample::FollowCamera) {
            scene.trampleCenter = {camera.position.x, camera.position.z};
            scene.trampleRadius = 1.5f;
        }

        // CPU-side preparation (animation, batch builds, uploads) before any draw pass.
        grassRenderer.prepare(grass, scene, app.render, cpu, gpu);

        // Stand-in for the rest of a game's frame. It runs after the simulation readback, where
        // gameplay would consume the result, and before the draws are submitted. Only the time
        // matters here: a spin loop does not compete for caches like real code would.
        if (app.gameLogicMs > 0.0f) {
            CpuScope scope(cpu, "Game logic (simulated)");
            const Clock::time_point until =
                Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                   std::chrono::duration<double, std::milli>(app.gameLogicMs));
            while (Clock::now() < until) {
            }
        }

        glBindFramebuffer(GL_FRAMEBUFFER, sceneTarget.framebuffer());
        glViewport(0, 0, internalWidth, internalHeight);
        glEnable(GL_DEPTH_TEST);
        glClearColor(scene.fogColor.r, scene.fogColor.g, scene.fogColor.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        {
            GpuScope g(gpu, "Ground");
            CpuScope c(cpu, "Ground submit");
            ground.draw(scene, std::fmax(grass.fieldSize(), 10.0f) * 0.75f);
        }
        {
            GpuScope g(gpu, "Grass");
            grassRenderer.submit(grass, scene, app.render, cpu);
        }
        {
            GpuScope g(gpu, "Resolve to window");
            glBlitNamedFramebuffer(sceneTarget.framebuffer(), 0, 0, 0, internalWidth, internalHeight,
                                   0, 0, fbWidth, fbHeight, GL_COLOR_BUFFER_BIT,
                                   internalWidth == fbWidth ? GL_NEAREST : GL_LINEAR);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, fbWidth, fbHeight);
        }
        {
            GpuScope g(gpu, "UI");
            CpuScope c(cpu, "UI render");
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }
        gpu.endFrame();
        frameFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

        const Clock::time_point swapStart = Clock::now();
        report.cpuWorkMs = msBetween(workStart, swapStart);
        glfwSwapBuffers(window);
        report.swapMs = msBetween(swapStart, Clock::now());
        ++frameIndex;
    }

    glFinish();
    for (GLsync fence : frameFences) {
        if (fence) glDeleteSync(fence);
    }
    gpu.destroy();
    sceneTarget.destroy();
    grassRenderer.destroy();
    grass.destroy();
    ground.destroy();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return app.regenerateFailed ? 2 : 0;
}
