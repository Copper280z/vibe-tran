# NastranSolver

A Nastran-compatible finite element solver written in modern C++20.

## Architecture

```
nastran_solver/
├── include/
│   ├── core/           # Matrix, DOF, Mesh data structures
│   ├── elements/       # Element formulations (CQUAD4, CTRIA3, CHEXA, CTETRA)
│   ├── io/             # BDF parser, F06 writer
│   ├── solver/         # Solver backends and linear static analysis
│   └── utils/          # Logging, error handling
├── src/                # Implementations
├── tests/
│   ├── unit/           # Unit tests per module
│   └── integration/    # End-to-end analysis cases
└── third_party/        # Eigen (header-only)
```

## Design Principles

- **Separation of concerns**: Parser, FE model, solver, and output are independent layers
- **Extensibility**: New element types implement `ElementBase`; new solvers implement `SolverBackend`
- **Modern C++20**: spans, ranges, strong types, error values over exceptions throughout

## Supported Features

- **Elements**: CQUAD4, CTRIA3, CHEXA8, CTETRA4
- **Materials**: MAT1 (isotropic)
- **Properties**: PSHELL, PSOLID
- **Loads**: FORCE, MOMENT, TEMP (thermal)
- **Constraints**: SPC, SPC1
- **Solution**: SOL 101 (Linear Static)

## Building

```bash
meson setup build
cd build
ninja
meson test
```

Optional backends are detected automatically at configure time.

## Usage

```
nastran_solver [--backend=<cpu|vulkan|cuda>] [--cuda-single-precision] <input.bdf> [output.f06]
```

If no output path is given, it defaults to `<input>.f06`.

## Solver Backends

### CPU (default)

Eigen sparse Cholesky. Always available, no extra dependencies.

```bash
nastran_solver --backend=cpu model.bdf
```

### CUDA — cuDSS (recommended for large models)

Uses [NVIDIA cuDSS](https://developer.nvidia.com/cudss) for GPU-resident sparse
direct solve. Requires CUDA toolkit ≥ 11 and cuDSS ≥ 0.7.

**Solver strategy:**
1. Sparse Cholesky (`CUDSS_MTYPE_SPD`) — optimal for SPD FEM stiffness matrices
2. Sparse LU (`CUDSS_MTYPE_GENERAL`) fallback for non-SPD or ill-conditioned matrices

**Memory management:**
- After symbolic analysis, cuDSS memory estimates are logged (device/host stable and peak).
- If the estimated device peak exceeds 85% of free GPU memory, hybrid host/device memory
  mode is enabled automatically so factor storage spills to host RAM.
- If analysis itself fails with an allocation error, hybrid mode is retried before giving up.

```bash
nastran_solver --backend=cuda model.bdf
```

**Single-precision mode** halves GPU memory usage by downcasting to float32 before the
solve and upcasting the result. Useful for very large models that exhaust device memory
even with hybrid mode:

```bash
nastran_solver --backend=cuda --cuda-single-precision model.bdf
```

#### Installing cuDSS (Ubuntu 24.04)

```bash
# Add the local repo and install
sudo dpkg -i cudss-local-repo-ubuntu2404-*.deb
sudo apt update
sudo apt install libcudss0-cuda-12 libcudss0-dev-cuda-12
```

### Vulkan

PCG iterative solver using Vulkan compute shaders. Requires Vulkan SDK, `glslc`, and `xxd`.

```bash
nastran_solver --backend=vulkan model.bdf
```

## Dependencies

| Dependency | Required | Purpose |
|---|---|---|
| C++20 compiler (GCC 11+, Clang 13+) | Yes | |
| Eigen 3.4 | Yes | CPU sparse Cholesky, matrix assembly |
| TBB | Yes | Parallel element assembly |
| Google Test | Yes (tests) | Fetched by Meson WrapDB |
| CUDA toolkit ≥ 11 + cuDSS ≥ 0.7 | No | CUDA backend |
| Vulkan SDK + glslc + xxd | No | Vulkan backend |
