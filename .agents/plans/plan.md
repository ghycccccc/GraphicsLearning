# Instanced Rendering Testbed — Plan

## Goal
A local C++ renderer that draws 10k–1M+ instances (a grass field is the main scene) with every
optimization behind a runtime toggle. It measures each toggle's effect so that you can see
*where the bottleneck moves* as you enable more of them.

Key idea: instancing reduces **CPU/driver submission cost**. It does not reduce GPU vertex or pixel
work. Once submission is cheap, the problem becomes deciding *what* to draw (culling, LOD) and
*who owns the data* (CPU or GPU).

## Stack
- **OpenGL 4.6 core + DSA**, C++17, CMake
- GLFW (window), glad (loader), GLM (math), Dear ImGui (toggles), ImPlot (live graphs), Tracy (optional profiler)
- Python + matplotlib for offline plots from CSV
- Later, optional: port the GPU-driven path to Vulkan/D3D12 to get Nsight GPU Trace and PIX hardware counters

Machine notes: RTX 2080 Super Max-Q (Turing) on an Optimus laptop.
- Export `NvOptimusEnablement = 1` so the app runs on the NVIDIA GPU.
- Turn vsync off when measuring.
- Keep the laptop plugged in and use a fixed power mode.
- `cl.exe` is not on PATH, so confirm that a VS/Build Tools install exists before building.

## Scenes
- **Grass field** (primary): blades from 10k to 1M+, with wind, alpha-test/opaque variants, and LODs.
- **Rocks/asteroids** (secondary): heavier meshes, so draw-call cost shows up with non-trivial vertex counts.
- **Particles** (optional): per-frame simulation, plus sorting for transparency.

## Optimization toggles

### A. Submission (CPU/driver)
0. One draw per instance, per-draw uniforms (baseline); variants with redundant state changes and with sorted state
1. Static batching (pre-merged VBO)
2. Dynamic batching (CPU transforms into one buffer each frame)
3. Instancing: per-instance attributes (`glVertexAttribDivisor`) vs SSBO + `gl_InstanceID`
4. Instance data layout: `mat4` (64 B) vs packed (~16 B, half floats)
5. Upload path: `glBufferSubData` / orphaning / persistent-mapped triple-buffered ring with fences

### B. Data residency & simulation (compute + GPU-persistent data)
A separate toggle *group*, because it changes who owns the data. Every other group depends on it.
- **B1. CPU-authoritative:** the CPU simulates (wind/particles) and uploads all instance data every frame.
- **B2. GPU-persistent:** instances are generated **once** on the GPU and stored in SSBOs. A compute shader
  simulates them in place each frame. The CPU uploads only a small per-frame UBO (time, camera, wind params).
- **B3. Vertex-shader-only animation:** static persistent data; wind is computed procedurally in the VS
  with no simulation pass. This is the cheapest option when no state needs to persist.

Do not confuse B2 with the *persistent-mapped buffer* in A5. A5 uploads CPU data faster.
B2 removes the upload entirely.

Why B2 is worth its own toggle:
- Without it, GPU culling (C2) still pays for a full upload each frame, so the payoff of GPU-driven
  rendering only shows up with B2 enabled.
- It moves cost from PCIe bandwidth and CPU time to GPU compute. The benchmark should show the crossover
  for both static and animated scenes.
- It adds a cost of its own: anything the CPU needs back (collision, interaction) requires a readback,
  which stalls or adds latency. Demo the GPU-native alternative: a "trample" texture written by the
  player and sampled by the sim.

### C. Visibility
1. CPU frustum culling: per instance → per chunk (grid/BVH) → multithreaded
2. **GPU frustum culling** (compute): writes visible instances to a buffer via an atomic counter, then
   `glDrawElementsIndirect`. Include a deliberately bad readback-the-count option so the sync stall is visible.
3. LOD selection in compute + `glMultiDrawElementsIndirect` (full blade / simplified / impostor card);
   compaction by atomics vs prefix sum
4. **Hi-Z occlusion culling**: depth pyramid built from the previous frame; two-pass culling as the advanced option

### D. Pixel cost
1. Alpha-tested quads (`discard`) vs opaque geometric blades (effect on early-Z)
2. Depth prepass + `GL_EQUAL` main pass
3. Front-to-back sorting
4. Distance-based geometry reduction (fewer tiny triangles, less quad overdraw)

### E. Advanced
1. Several blades per instance (fixes vertex-warp waste with tiny meshes)
2. Mesh shaders (`GL_NV_mesh_shader`, task + mesh) that generate blades directly
3. Bitonic sort in compute for transparent particles

Dependencies are enforced in the UI:
- C2–C4 require the indirect draw path.
- C4 requires the previous frame's depth.
- C2 is meaningful mainly with B2.
- E2 replaces the vertex pipeline path.

## Measurement

### Live (ImGui/ImPlot)
- Frame time in **ms** (not FPS), with **CPU and GPU frame time shown separately** to tell CPU-bound from GPU-bound.
- GPU time per pass, from `GL_TIMESTAMP` queries in a 3–4 frame ring so reads never stall.
  Passes: sim, cull, Hi-Z, prepass, main, post.
- CPU time per stage: sim, cull, upload, GL submission (driver overhead).
- Counters:
  - draw calls and state changes
  - instances submitted / visible / per LOD
  - triangles submitted
  - **bytes uploaded per frame**
  - GPU memory used
- Pipeline statistics (`GL_ARB_pipeline_statistics_query`): VS invocations, clipping primitives, FS invocations.
  - Overdraw ratio = FS invocations ÷ screen pixels.
  - `GL_SAMPLES_PASSED` to cross-check culling.
- Frame-time distribution: average, p95, p99, 1% low, and a histogram (stalls appear as spikes).

### Benchmark mode
- Fixed scripted camera path with segments for "all visible", "mostly occluded" and "edge of field"; warm-up frames first.
- Sweep instance counts (1k → 1M) × chosen toggle presets → CSV.
- **Budget capacity**: for each technique, ramp N until the frame time exceeds the budget and record the
  max N that fits. This is the headline number and comparable across GPUs (dGPU vs iGPU).
- Python script: log-log cost-vs-N curves to find crossover points (e.g. where GPU culling starts to pay off,
  where instancing stops helping because the GPU is the bottleneck).

### Debug views
- Overdraw heatmap
- Coloring by LOD and by culled/visible state
- **Frozen culling camera**: fly outside the frustum to inspect what was culled
- Hi-Z mip viewer
- Optional "sync to measure" mode (`glFinish`)

### External tools
RenderDoc (frame/buffer/indirect-args inspection), Nsight Graphics (frame debugger), Tracy (CPU+GPU timeline).

## Milestones
1. **Skeleton**: window, camera, ImGui, grass blade mesh, CPU+GPU timers, counters panel.
1.5. **Stress controls** (instanced path alone barely loads the RTX 2080 at 2M blades, so the GPU-side
   optimizations need a GPU-bound starting point):
   - instance cap raised to 16M
   - blade segments slider (1–16 levels → 1–31 tris)
   - render scale 0.25–4× via an offscreen FBO + blit (fill-rate stress; the depth texture is reused later by Hi-Z)
   - frame-budget line (30/60/120/144 fps) with headroom and % of frames over budget
   - `InstancingLab_iGPU` build without the Optimus hint, to run on the Intel UHD as a low-end target
   - later stress hooks: shadow-map pass (M4), alpha-tested cards (M5)
2. **Submission** (done): A0 (+ redundant-state variant), A1, A2, A3 (attributes / SSBO), A5 (SubData /
   orphaning / persistent ring). B1 "CPU per-blade wind" landed here because uploads need per-frame CPU data.
   - Bend is now a world-space vector in the instance data, so CPU wind and VS wind share one layout.
   - Frame pacing: max 2 frames in flight via a fence at frame start. Time waiting on it (+ swap) is the
     GPU-bound signal; GPU timestamp spans include idle gaps inside a pass, so they can't decide this alone.
   - GPU work = sum of top-level pass timestamps; CPU prep (animate/build/upload) runs before any pass.
   - Benchmark CLI (`--path --animation --upload --bench-frames --csv`) + `tools/bench.py run|plot`; a minimal
     version of M6, without the camera path.
   - A4 (packed 16 B layout) moved to M3, where bandwidth is the topic.
3. **Residency**: B2 compute + GPU-persistent data, B3 is the current VS wind; A4 packed layout.
   Detailed plan: [m3-residency.md](m3-residency.md) (stateful spring sim, CPU 1T/MT vs compute,
   static/dynamic data split, 16 B layout, trample interactor, readback stall demo).
4. **Culling** (done): C1 (CPU per blade 1 thread / all cores, CPU chunks + MDI) → C2 (GPU compute + indirect,
   global / workgroup / prefix-sum compaction) → C3 (3 LODs, one multi-draw); frozen culling camera, color by
   gl_DrawID, readback anti-pattern. Details, reading guide and results: [m4-culling.md](m4-culling.md).
   Shadow-map stress pass deferred (M4.5 or with M5).
5. **Pixel cost**: D1–D4, pipeline statistics, overdraw heatmap.
6. **Benchmark mode**: camera path, sweeps, CSV, Python plots.
7. **Occlusion & advanced**: C4 Hi-Z, E1, E2 mesh shaders, E3 particles.
8. Optional: port the GPU-driven path to Vulkan/D3D12 to get hardware counters.

## Interview takeaways to verify with data
- Instancing removes CPU/driver cost but does nothing once the GPU is the bottleneck.
- Tiny meshes instance poorly (warp waste); merging several blades per instance fixes it.
- GPU culling pays off only if the data is already GPU-resident and never read back.
- For grass, alpha-test + `discard` + overdraw often cost more than vertex work; early-Z and LOD matter more than draw count.
