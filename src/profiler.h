#pragma once
#include <GL/glew.h>

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

// GPU pass timing with GL_TIMESTAMP queries. Each frame writes into one of kFrames
// query slots; a slot is read back only when it comes round again, so reading never
// stalls the pipeline. Results are therefore kFrames-1 frames old.
class GpuProfiler {
public:
    static constexpr int kFrames = 5;
    static constexpr int kMaxScopes = 32;

    struct Result {
        const char* name;
        double ms;    // latest sample
        double avgMs; // exponential moving average
        int depth;
    };

    void init();
    void destroy();

    void beginFrame();
    int beginScope(const char* name);
    void endScope(int id);
    void endFrame();

    const std::vector<Result>& results() const { return results_; }
    double frameMs() const { return results_.empty() ? 0.0 : results_[0].ms; }
    int missedReadbacks() const { return missed_; }

private:
    struct Slot {
        GLuint queries[kMaxScopes * 2] = {};
        const char* names[kMaxScopes] = {};
        int depth[kMaxScopes] = {};
        int count = 0;
        bool pending = false;
    };

    void collect(Slot& slot);

    Slot slots_[kFrames];
    int current_ = 0;
    int depth_ = 0;
    int frameScope_ = -1;
    int missed_ = 0;
    std::vector<Result> results_;
    std::unordered_map<std::string, double> ema_;
};

struct GpuScope {
    GpuScope(GpuProfiler& p, const char* name) : profiler(p), id(p.beginScope(name)) {}
    ~GpuScope() { profiler.endScope(id); }
    GpuProfiler& profiler;
    int id;
};

// Flat CPU stage timings for the current frame; the UI reads the previous frame's.
class CpuProfiler {
public:
    struct Result {
        const char* name;
        double ms;
        double avgMs;
    };

    void beginFrame();
    void record(const char* name, double ms);
    const std::vector<Result>& results() const { return last_; }

private:
    std::vector<Result> current_, last_;
    std::unordered_map<std::string, double> ema_;
};

struct CpuScope {
    using Clock = std::chrono::steady_clock;
    CpuScope(CpuProfiler& p, const char* n) : profiler(p), name(n), start(Clock::now()) {}
    ~CpuScope() {
        profiler.record(name, std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    CpuProfiler& profiler;
    const char* name;
    Clock::time_point start;
};

// Ring buffer of per-frame times with percentile statistics.
class FrameHistory {
public:
    static constexpr int kCapacity = 1024;
    enum Series { Frame, Cpu, Gpu, kSeriesCount };

    struct Stats {
        float avg = 0, p50 = 0, p95 = 0, p99 = 0, max = 0;
        float worst1PctAvg = 0; // mean of the slowest 1% of frames ("1% low")
    };

    void push(float frameMs, float cpuMs, float gpuMs);
    void clear() { head_ = size_ = 0; }
    Stats stats(Series series, int window) const;
    // Fraction of the last `window` samples above `thresholdMs`.
    float fractionAbove(Series series, int window, float thresholdMs) const;

    const float* data(Series series) const { return data_[series]; }
    int size() const { return size_; }
    int offset() const { return size_ < kCapacity ? 0 : head_; }

private:
    float data_[kSeriesCount][kCapacity] = {};
    int head_ = 0;
    int size_ = 0;
};
