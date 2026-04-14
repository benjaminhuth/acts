# Upstream PR Plan

Analysis of `git diff upstream/main` on `tmp/g300-merged-main`.

---

## PR 1+2: Runtime CPU/GPU selection (no new dependencies)

**Goal:** Replace compile-time `ACTS_GNN_CPUONLY` flag with a runtime `useCuda` config field
throughout the stack. Enables running the existing ONNX/ModuleMap pipeline on CPU without
recompiling.

**Files:**

| File | Change |
|------|--------|
| `Plugins/Gnn/include/ActsPlugins/Gnn/OnnxEdgeClassifier.hpp` | Add `bool useCuda = true` to `Config`; forward-declare `Ort::MemoryInfo` |
| `Plugins/Gnn/src/OnnxEdgeClassifier.cpp` | Use `useCuda` flag instead of `#ifdef ACTS_GNN_CPUONLY`; use templated `CreateTensor<T>`; add shape debug logging; throw on missing edge features |
| `Examples/Algorithms/TrackFindingGnn/include/.../TrackFindingAlgorithmGnn.hpp` | Add `bool useCuda = true` to `Config` |
| `Examples/Algorithms/TrackFindingGnn/src/TrackFindingAlgorithmGnn.cpp` | Replace `#ifdef ACTS_GNN_CPUONLY` with `m_cfg.useCuda ? Device::Cuda(0) : Device::Cpu()` |
| `Python/Examples/src/plugins/Gnn.cpp` | Expose `useCuda` in `TrackFindingAlgorithmGnn` Python binding |
| `Python/Examples/python/reconstruction.py` | Thread `useCuda` parameter through `addGnn()` |
| `Python/Plugins/src/Gnn.cpp` | Expose `useCuda` in `OnnxEdgeClassifier` binding |
| `Examples/Scripts/Python/gnn.py` | Set `useCuda = False` for metric-learning ONNX path (CPU-only use case) |
| `Examples/Scripts/Python/gnn_module_map_odd.py` | Set `useCuda = True` explicitly |

**Notes:**
- No new dependencies.
- Removes the `ACTS_GNN_CPUONLY` compile-time define from the runtime path entirely.

---

## PR 3: `ModuleMapCpu` — CPU graph construction + `EdgeLayerConnector`

**Goal:** Port the CUDA module-map graph construction to CPU using the existing
`ModuleMapGraph::CPU` library (already a dependency when `ACTS_GNN_ENABLE_MODULEMAP=ON`).
Also decouple the CMake build so `ModuleMapGraph::GPU` is only required when
`ACTS_GNN_ENABLE_CUDA=ON`. Add `EdgeLayerConnector` as a CUDA track-building alternative
using MMG's `CUDA_edge_layer_connector`.

**Files:**

| File | Change |
|------|--------|
| `Plugins/Gnn/include/ActsPlugins/Gnn/ModuleMapCpu.hpp` | New class: CPU doublet/triplet graph construction |
| `Plugins/Gnn/src/ModuleMapCpu.cpp` | New impl: doublet loop, triplet loop, edge feature computation |
| `Plugins/Gnn/include/ActsPlugins/Gnn/EdgeLayerConnector.hpp` | New class: CUDA track building via MMG `CUDA_edge_layer_connector` |
| `Plugins/Gnn/src/EdgeLayerConnector.cu` | New CUDA impl |
| `Plugins/Gnn/CMakeLists.txt` | Decouple `ModuleMapGraph::CPU` / `::GPU`; add `ModuleMapCpu.cpp`; guard `EdgeLayerConnector.cu` on `ACTS_GNN_ENABLE_CUDA` |
| `thirdparty/ModuleMapGraph/CMakeLists.txt` | Apply patch on fetch |
| `thirdparty/ModuleMapGraph/remove_unused_members.patch` | Remove two unused members from `CUDA_edge_layer_connector` (fixes ODR violations) |
| `Tests/UnitTests/Plugins/Gnn/ModuleMapCpuTests.cpp` | Unit tests: invalid path throws, edge feature values, zero-dr, phi wrapping |
| `Tests/UnitTests/Plugins/Gnn/CMakeLists.txt` | Register `ModuleMapCpu` test under `ACTS_GNN_ENABLE_MODULEMAP` guard |
| `Python/Plugins/src/Gnn.cpp` | Expose `ModuleMapCpu` (behind `ACTS_GNN_WITH_MODULEMAP`) and `EdgeLayerConnector` (behind `ACTS_GNN_WITH_MODULEMAP && ACTS_GNN_WITH_CUDA`) |
| `Examples/Scripts/Python/gnn4itk_example.py` | Add `--cpuOnly` flag wiring `ModuleMapCpu` + `BoostTrackBuilding`; add `--useEdgeLayerConnector` flag; add `--bufferEvents` / `BufferedReader` support |

**Notes:**
- `ModuleMapGraph::CPU` is **not** a new dependency — it was already pulled in by
  `ACTS_GNN_ENABLE_MODULEMAP`. The CMake change just stops requiring the GPU half unconditionally.
- `EdgeLayerConnector` is fully guarded by `ACTS_GNN_WITH_CUDA`; CPU-only builds are unaffected.
- The patch to `ModuleMapGraph` is required to fix ODR violations when compiling
  `EdgeLayerConnector.cu`.
- **No tests yet for `EdgeLayerConnector`** — worth flagging in the PR description.

---

## PR 4: `SofieEdgeClassifier` — ROOT SOFIE inference backend

**Goal:** Add a new edge classifier backed by ROOT's SOFIE (System for On-the-Fly Inference
Execution), fetching a new thirdparty library `sofie-atlas-tracking`.

**Files:**

| File | Change |
|------|--------|
| `Plugins/Gnn/include/ActsPlugins/Gnn/SofieEdgeClassifier.hpp` | New class |
| `Plugins/Gnn/src/SofieEdgeClassifier.cpp` | Impl: tries small model variant first, falls back to large |
| `CMakeLists.txt` | Add `ACTS_GNN_ENABLE_SOFIE` option; `add_subdirectory(thirdparty/SofieGnn)` |
| `Plugins/Gnn/CMakeLists.txt` | Link `sofie_gnn`; compile definition `ACTS_GNN_SOFIE_BACKEND` |
| `cmake/ActsExternSources.cmake` | Add `ACTS_SOFIEGNN_SOURCE` FetchContent variable |
| `thirdparty/SofieGnn/CMakeLists.txt` | New thirdparty integration (FetchContent from `sanjibansg/sofie-atlas-tracking`) |
| `Python/Plugins/src/Gnn.cpp` | Expose `SofieEdgeClassifier` behind `ACTS_GNN_SOFIE_BACKEND` |
| `Examples/Scripts/Python/gnn4itk_example.py` | Add `.dat` model path → `SofieEdgeClassifier` branch |

**Notes / concerns for upstream review:**
- Introduces a new thirdparty dependency (`sofie-atlas-tracking` at a pinned git hash).
- `SofieEdgeClassifier.cpp` directly `#include`s generated `.hxx` headers
  (`gnn_large_dynamic.hxx`, `gnn_small_dynamic.hxx`) and hardcodes two named namespaces
  (`TMVA_SOFIE_gnn`, `TMVA_SOFIE_gnn_large`). This is fragile — upstream reviewers will
  likely push back. The model-variant selection should probably be made configurable rather
  than tried sequentially.
- No unit tests.

---

## Not for upstreaming

| Item | Reason |
|------|--------|
| `version_number`, `.zenodo.json`, `CITATION.cff` | Release artifacts; upstream manages these |
| Untracked log/script files (`*.log`, `run_*.sh`, etc.) | Local debugging artifacts |
