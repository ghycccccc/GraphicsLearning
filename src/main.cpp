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
#include "profiler.h"
#include "render_target.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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
    int budgetIndex = 2;
    bool vsync = false;
    float windStrength = 0.35f;
    float windAngle = 0.6f;
    float fogDensity = 0.012f;
    int statsWindow = 240;
    bool resetStats = false;
    int warmupFrames = 0; // frames left to skip after a reset
};

// Everything the UI reads, captured at the end of the previous frame.
struct FrameReport {
    FrameCounters counters;
    double cpuWorkMs = 0.0;
    double swapMs = 0.0;
    uint64_t vramTotalKb = 0;
    uint64_t vramAvailableKb = 0;
    int internalWidth = 0;
    int internalHeight = 0;
};

// Command-line overrides, e.g. --instances 4000000 --segments 8 --scale 2
void parseArgs(int argc, char** argv, AppState& app) {
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        const char* value = argv[i + 1];
        if (key == "--instances") app.instanceCount = std::clamp(std::atoi(value), 1, kMaxInstances);
        else if (key == "--density") app.density = float(std::atof(value));
        else if (key == "--segments") app.segments = std::clamp(std::atoi(value), 1, GrassField::kMaxSegments);
        else if (key == "--scale") app.renderScale = std::clamp(float(std::atof(value)), 0.25f, 4.0f);
        else std::fprintf(stderr, "Unknown argument %s\n", key.c_str());
    }
}

std::string withSeparators(uint64_t value) {
    std::string digits = std::to_string(value);
    for (int i = int(digits.size()) - 3; i > 0; i -= 3) digits.insert(size_t(i), ",");
    return digits;
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

void drawUi(AppState& app, FlyCamera& camera, const GrassField& grass, const GpuProfiler& gpu,
            const CpuProfiler& cpu, const FrameHistory& history, const FrameReport& report,
            const char* renderer) {
    ImGui::SetNextWindowPos({10, 10}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({440, 860}, ImGuiCond_FirstUseEver);
    ImGui::Begin("Instancing Lab");

    ImGui::TextDisabled("%s", renderer);
#ifdef LAB_INTEGRATED_GPU
    ImGui::TextColored({0.5f, 0.8f, 1.0f, 1.0f}, "iGPU build (no discrete GPU hint)");
#else
    if (!std::strstr(renderer, "NVIDIA") && !std::strstr(renderer, "AMD") &&
        !std::strstr(renderer, "Radeon")) {
        ImGui::TextColored({1.0f, 0.45f, 0.3f, 1.0f}, "Warning: not running on the discrete GPU");
    }
#endif

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

    if (ImGui::CollapsingHeader("Stress", ImGuiTreeNodeFlags_DefaultOpen)) {
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

        if (ImGui::BeginTable("timing", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            for (const char* h : {"ms", "avg", "p50", "p95", "p99", "max"})
                ImGui::TableSetupColumn(h);
            ImGui::TableHeadersRow();
            statsRow("Frame", frame);
            statsRow("CPU work", cpuStats);
            statsRow("GPU work", gpuStats);
            ImGui::EndTable();
        }
        ImGui::Text("avg %.0f fps   1%% low %.0f fps   swap %.2f ms", 1000.0f / frame.avg,
                    1000.0f / frame.worst1PctAvg, report.swapMs);

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

        // Whichever side's busy time is closest to the frame interval is the limiter.
        const float cpuLoad = cpuStats.avg / frame.avg;
        const float gpuLoad = gpuStats.avg / frame.avg;
        // A saturated GPU wins: once the driver queue is full the CPU blocks inside GL calls,
        // so CPU work then also reads close to 100% without the CPU being the limiter.
        const char* verdict = "Neither saturated (sync / latency bound)";
        if (app.vsync && std::fabs(frame.avg - 1000.0f / 60.0f) < 1.0f) verdict = "VSync-limited";
        else if (gpuLoad > 0.85f) verdict = "GPU-bound";
        else if (cpuLoad > 0.85f) verdict = "CPU-bound";
        ImGui::Text("Verdict: %s", verdict);
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset stats")) app.resetStats = true;
        if (gpuLoad > 0.85f && cpuLoad > 0.85f) {
            ImGui::TextDisabled("CPU work includes time blocked in the driver waiting on the GPU");
        }
        ImGui::ProgressBar(cpuLoad, {-1, 0}, ("CPU busy " + std::to_string(int(cpuLoad * 100)) + "%").c_str());
        ImGui::ProgressBar(gpuLoad, {-1, 0}, ("GPU busy " + std::to_string(int(gpuLoad * 100)) + "%").c_str());

        if (ImPlot::BeginPlot("##frametimes", {-1, 190})) {
            ImPlot::SetupAxes(nullptr, "ms", ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, FrameHistory::kCapacity, ImGuiCond_Always);
            ImPlot::SetupLegend(ImPlotLocation_NorthWest);
            const int n = history.size(), off = history.offset();
            ImPlot::PlotLine("Frame", history.data(FrameHistory::Frame), n, 1.0, 0.0, 0, off);
            ImPlot::PlotLine("CPU", history.data(FrameHistory::Cpu), n, 1.0, 0.0, 0, off);
            ImPlot::PlotLine("GPU", history.data(FrameHistory::Gpu), n, 1.0, 0.0, 0, off);
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
    Ground ground;
    if (!grass.init() || !ground.init()) return 1;

    GpuProfiler gpu;
    gpu.init();
    CpuProfiler cpu;
    FrameHistory history;
    FlyCamera camera;
    FrameReport report;
    RenderTarget sceneTarget;

    const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    std::printf("Renderer: %s\nVersion:  %s\n", renderer,
                reinterpret_cast<const char*>(glGetString(GL_VERSION)));
    const bool hasNvMemInfo = GLEW_NVX_gpu_memory_info;

    const SceneParams defaults;
    glEnable(GL_DEPTH_TEST);
    const Clock::time_point startTime = Clock::now();
    Clock::time_point prevFrameStart = startTime;
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

        // Frame interval is measured start-to-start, so it includes swap and any waiting.
        const Clock::time_point frameStart = Clock::now();
        const double frameMs = msBetween(prevFrameStart, frameStart);
        prevFrameStart = frameStart;
        if (app.warmupFrames > 0) {
            --app.warmupFrames;
        } else if (frameIndex > 0) {
            history.push(float(frameMs), float(report.cpuWorkMs), float(gpu.frameMs()));
        }

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
            drawUi(app, camera, grass, gpu, cpu, history, report, renderer);
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
        }
        if (app.segments != grass.segments()) {
            grass.setSegments(app.segments);
            app.resetStats = true;
        }
        if (app.resetStats) {
            // Drop samples from the old configuration, including in-flight GPU timings.
            history.clear();
            app.warmupFrames = GpuProfiler::kFrames + 1;
            app.resetStats = false;
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
            CpuScope c(cpu, "Grass submit");
            grass.draw(scene);
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

        const Clock::time_point swapStart = Clock::now();
        report.cpuWorkMs = msBetween(frameStart, swapStart);
        glfwSwapBuffers(window); // blocks when the driver's frame queue is full (GPU-bound)
        report.swapMs = msBetween(swapStart, Clock::now());
        ++frameIndex;
    }

    gpu.destroy();
    sceneTarget.destroy();
    grass.destroy();
    ground.destroy();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
