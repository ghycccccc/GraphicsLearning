#include "profiler.h"

#include <algorithm>

namespace {
constexpr double kEmaAlpha = 0.05;

double updateEma(std::unordered_map<std::string, double>& ema, const char* name, double ms) {
    auto [it, inserted] = ema.try_emplace(name, ms);
    if (!inserted) it->second += (ms - it->second) * kEmaAlpha;
    return it->second;
}
} // namespace

// --- GpuProfiler -------------------------------------------------------------

void GpuProfiler::init() {
    for (Slot& slot : slots_) glCreateQueries(GL_TIMESTAMP, kMaxScopes * 2, slot.queries);
}

void GpuProfiler::destroy() {
    for (Slot& slot : slots_) glDeleteQueries(kMaxScopes * 2, slot.queries);
}

void GpuProfiler::beginFrame() {
    Slot& slot = slots_[current_];
    if (slot.pending) collect(slot);
    slot.count = 0;
    slot.pending = false;
    depth_ = 0;
    frameScope_ = beginScope("GPU frame");
}

int GpuProfiler::beginScope(const char* name) {
    Slot& slot = slots_[current_];
    if (slot.count >= kMaxScopes) return -1;
    const int id = slot.count++;
    slot.names[id] = name;
    slot.depth[id] = depth_++;
    glQueryCounter(slot.queries[id * 2], GL_TIMESTAMP);
    return id;
}

void GpuProfiler::endScope(int id) {
    if (id < 0) return;
    --depth_;
    glQueryCounter(slots_[current_].queries[id * 2 + 1], GL_TIMESTAMP);
}

void GpuProfiler::endFrame() {
    endScope(frameScope_);
    slots_[current_].pending = true;
    current_ = (current_ + 1) % kFrames;
}

void GpuProfiler::collect(Slot& slot) {
    // The frame scope's end timestamp is the last query issued; if it is not ready,
    // reading would block, so drop this sample instead.
    GLint available = GL_FALSE;
    glGetQueryObjectiv(slot.queries[1], GL_QUERY_RESULT_AVAILABLE, &available);
    if (!available) {
        ++missed_;
        return;
    }
    results_.clear();
    for (int i = 0; i < slot.count; ++i) {
        GLuint64 begin = 0, end = 0;
        glGetQueryObjectui64v(slot.queries[i * 2], GL_QUERY_RESULT, &begin);
        glGetQueryObjectui64v(slot.queries[i * 2 + 1], GL_QUERY_RESULT, &end);
        const double ms = double(end - begin) * 1e-6;
        results_.push_back({slot.names[i], ms, updateEma(ema_, slot.names[i], ms), slot.depth[i]});
    }
}

// --- CpuProfiler -------------------------------------------------------------

void CpuProfiler::beginFrame() {
    last_.swap(current_);
    current_.clear();
}

void CpuProfiler::record(const char* name, double ms) {
    current_.push_back({name, ms, updateEma(ema_, name, ms)});
}

// --- FrameHistory ------------------------------------------------------------

void FrameHistory::push(float frameMs, float cpuMs, float gpuMs) {
    data_[Frame][head_] = frameMs;
    data_[Cpu][head_] = cpuMs;
    data_[Gpu][head_] = gpuMs;
    head_ = (head_ + 1) % kCapacity;
    size_ = std::min(size_ + 1, kCapacity);
}

FrameHistory::Stats FrameHistory::stats(Series series, int window) const {
    Stats s;
    const int n = std::min(window, size_);
    if (n == 0) return s;

    float values[kCapacity];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        values[i] = data_[series][(head_ - 1 - i + kCapacity) % kCapacity];
        sum += values[i];
    }
    std::sort(values, values + n);

    auto percentile = [&](float p) { return values[std::min(n - 1, int(p * (n - 1) + 0.5f))]; };
    s.avg = float(sum / n);
    s.p50 = percentile(0.50f);
    s.p95 = percentile(0.95f);
    s.p99 = percentile(0.99f);
    s.max = values[n - 1];

    const int worst = std::max(1, n / 100);
    double worstSum = 0.0;
    for (int i = n - worst; i < n; ++i) worstSum += values[i];
    s.worst1PctAvg = float(worstSum / worst);
    return s;
}

float FrameHistory::fractionAbove(Series series, int window, float thresholdMs) const {
    const int n = std::min(window, size_);
    if (n == 0) return 0.0f;
    int above = 0;
    for (int i = 0; i < n; ++i) {
        if (data_[series][(head_ - 1 - i + kCapacity) % kCapacity] > thresholdMs) ++above;
    }
    return float(above) / float(n);
}
