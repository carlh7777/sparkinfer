# Qwen3-30B-A3B Q4_K_M — RTX 5090 verification

**Hardware:** NVIDIA GeForce RTX 5090 (sm_120, 32 GB GDDR7, ~1792 GB/s, driver 580.142), vast.ai
**Toolchain:** CUDA 13.0, gcc 13, CMake 3.28
**Model:** Qwen3-30B-A3B Q4_K_M GGUF (18.5 GiB, official `Qwen/Qwen3-30B-A3B-GGUF`)
**Setting:** single stream, batch = 1, greedy decode

## Result — runs end-to-end, correct, fits in 32 GB

| Check | Value |
|---|---|
| Build (CMake superbuild, sm_120) | ✓ clean |
| `ctest` | ✓ **5/5 pass** |
| compute-sanitizer (memcheck) | ✓ **0 errors** |
| VRAM resident (experts quantized) | **21.4 GB** |
| Decode throughput | **163.88 tok/s** (6.1 ms/token, n=128) |
| Correctness | "What is the capital of France?" → **"The capital of France is Paris."** ✓ |

## Cross-config note — 5090 vs PRO 6000 (not apples-to-apples)

The PRO 6000 measured **133–134 tok/s** (`qwen3-30b-a3b_q4km_pro6000.md`); the 5090 here hits **163.88**. The two runs differ in **both the GPU and the toolchain** (5090 / CUDA 13 vs PRO 6000 / CUDA 12.8), so the gap can't be cleanly attributed to one factor. Likely contributors, not yet isolated:

- **Core clock** — decode at bs=1 is latency-bound (vectorizing & norm-fusion both gave ~0), so it's sensitive to SM clock; the consumer RTX 5090 boosts higher than the PRO 6000 Server Edition at the same 1792 GB/s memory bandwidth.
- **CUDA 13 codegen / driver** — a newer `nvcc` + driver may schedule/emit the kernels better.

Isolating it would need the same CUDA toolkit on both cards (or clock-locked runs). Current read: probably a mix, with core clock the likely larger term for latency-bound decode — but the CUDA-version contribution is plausible and untested.

## Bugs found & fixed during 5090 bring-up

Building + testing on this second GPU with a newer toolchain surfaced three real issues a single-box history had hidden (all fixed in `sparkinfer`):

1. **CUDA 13 removed** `cudaDeviceProp::memoryClockRate` / `memoryBusWidth` → query via `cudaDeviceGetAttribute`.
2. **flash-decode scratch (`fa_*`) was NULL on the non-GGUF path** (allocated only in `load_gguf`) → moved to the constructor; caught by compute-sanitizer.
3. **top-level superbuild lacked `enable_testing()`** → `ctest` found no tests.

## Reproduce

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j
HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --local-dir models
./build/runtime/qwen3_gguf_bench models/Qwen3-30B-A3B-Q4_K_M.gguf 128
python3 runtime/tools/run_qwen3.py models/Qwen3-30B-A3B-Q4_K_M.gguf \
  ./build/runtime/qwen3_gguf_generate "What is the capital of France?" 16
```
