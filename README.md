# Vibestran

A finite element solver written in modern C++20 that attempts to be
Nastran/Mystran-compatible.

## Why

I was curious how feasible it would be to create something this complex with an AI agent. It's a well documented field, but it also requires that the math is precisely correct. I also wanted to experiment with seeing how fast a FEM solver could be when written in a language I understand (well, within the limits of one's ability to understand C++), and using GPU acceleration. I was prepared to do the benchmarking and optimization myself, but it turned out that Claude was quite capable with some guidance. 

Vibestran draws significant inspiration from the [Mystran](https://github.com/MYSTRANsolver/MYSTRAN)
project. If you're unfamiliar with the Mystran project you should go look at it first. It should be considered the reference for open source Nastran-compatible solvers. Without it, as an introduction to OSS FEM Solvers, there's zero chance I would have tried this.

The OpenJFEM project, which is another solver, written in Julia, was also an inspiration for trying this with an agent.

## Performance

More coming soon. 84k elem/170k node hex mesh model solves in 2.6sec on M2 macbook air, 3.6sec on an rtx3060 12gb+ryzen 5700, and 4.3s on the ryzen 5700 alone. 

## Architecture

```
vibestran/
├── include/
│   ├── core/           # Matrix, DOF, Mesh data structures
│   ├── elements/       # Shell, solid, beam, bush, spring, and lumped-mass elements
│   ├── io/             # BDF/INP parsers, F06/OP2/CSV writers
│   ├── solver/         # Solver backends and static/modal analysis
│   └── utils/          # Logging, error handling
├── src/                # Implementations
├── tests/
│   ├── unit/           # Unit tests per module
│   ├── integration/    # Integration tests with hand-calculated solutions
│   └── e2e/            # End-to-end analysis cases (BDF + expected JSON, run against vibestran binary)
└── third_party/        # Bundled third-party headers (for example Spectra)
```

## Design Principles

- **Separation of concerns**: Parser, FE model, solver, and output are independent layers
- **Extensibility**: New element types implement `ElementBase`; new solvers implement `SolverBackend`
- **Modern C++20**: spans, ranges, strong types, error values over exceptions throughout
- **Performance**: Hash maps are wonderful. Parallel assembly and sparse solves where available, with CUDA, Vulkan, and Apple Accelerate-backed sparse factorizations.

## Supported Features

- **Elements**: CQUAD4, CTRIA3, CHEXA8, CHEXA20, CTETRA4, CTETRA10, CPENTA6, CBAR, CBEAM, CBUSH, CELAS1, CELAS2, CMASS1, CMASS2
- **Materials**: MAT1 (isotropic)
- **Properties**: PSHELL, PSOLID, PBAR, PBARL (`ROD`, `TUBE`, `BAR`), PBEAM, PBUSH (`K` section), PELAS, PMASS
- **Loads**: FORCE, MOMENT, TEMP/TEMPD, PLOAD, PLOAD1 (CBAR/CBEAM), PLOAD2, PLOAD4, GRAV, ACCEL1
- **Constraints**: SPC, SPC1, MPC, RBE2, RBE3
- **Coordinate systems**: CORD1R/C/S, CORD2R/C/S
- **Input formats**: Nastran BDF (`.bdf`) and CalculiX/Abaqus (`.inp`, experimental)
- **Solutions**: SOL 101 (Linear Static), SOL 103 (Normal Modes / Modal Analysis)

Notable current limitations:

- `ACCEL` is parsed but not yet applied in the static solver.
- PLOAD1 support is limited to CBAR and CBEAM.
- Stress recovery is not yet implemented for CBAR, CBEAM, CBUSH, CELAS1/2, and CMASS1/2.

## SOL 103 — Normal Modes (Modal Analysis)

Compute natural frequencies and mode shapes using a generalized eigensolver. Activated
by `SOL 103` in the BDF case control deck.

```
vibestran model.bdf
```

Eigensolvers are selected via `--backend`:

| Backend | Eigensolver | Notes |
|---|---|---|
| `cpu` (default) | Spectra shift-and-invert + sparse direct solve | Uses Apple Accelerate on macOS when available, otherwise CHOLMOD/Eigen |
| `cuda` / `cuda-pcg` | Implicitly Restarted Lanczos (IRL) with Rayleigh-Ritz refinement | GPU-accelerated; requires CUDA + cuDSS |

On macOS, the CPU modal path prefers Apple Accelerate for the shift-invert
factorization and triangular solves used inside Spectra. The Accelerate sparse
ordering can be controlled with `VIBESTRAN_ACCELERATE_ORDER=metis|amd|default`;
the default is `metis`.

The CUDA eigensolver uses cuDSS for the shift-and-invert linear solves inside
the Lanczos iteration, then applies a Rayleigh-Ritz post-refinement step to
improve accuracy of the converged eigenpairs.

## Building

```bash
meson setup build
cd build
ninja
meson test
ninja cppcheck
```

Optional solver backends are detected automatically at configure time.

## Usage

```
vibestran [--backend=<cpu|cpu-pcg|vulkan|cuda|cuda-pcg|cuda-pcg-mixed>]
         [--cuda-precision=<fp32|fp64>] [--csv]
         [--log-file=<path>]
         <input.bdf|input.inp> [output.f06]
```

Both Nastran BDF (`.bdf`) and CalculiX/Abaqus (`.inp`) input files are supported.
The format is detected by file extension. If no output path is given, it defaults
to `<input>.f06`.

### CalculiX/Abaqus `.inp` support (experimental)

The `.inp` parser accepts standard CalculiX input files and maps them to the same
internal model used by the BDF parser. Supported keywords include `*NODE`,
`*ELEMENT`, `*MATERIAL`/`*ELASTIC`/`*DENSITY`/`*EXPANSION`, `*SOLID SECTION`,
`*SHELL SECTION`, `*BOUNDARY`, `*CLOAD`, `*TEMPERATURE`, `*STEP`/`*STATIC`/`*END STEP`,
and output request keywords (`*NODE FILE`, `*EL FILE`, `*NODE PRINT`, `*EL PRINT`).

Supported element types: C3D8, C3D4, C3D10, C3D6, S4, S3 (and their R variants).
C3D20 is not currently supported.

See `include/io/inp_parser.hpp` for full details on design decisions and limitations.

### Output files

| File | Format | Written when |
|---|---|---|
| `<stem>.f06` | F06 text | Always |
| `<stem>.op2` | OP2 binary | Always |
| `<stem>.node.csv` | Nodal results CSV | `--csv` flag or `PARAM,CSVOUT,YES` in BDF |
| `<stem>.elem.csv` | Element results CSV | `--csv` flag or `PARAM,CSVOUT,YES` in BDF |

## Solver Backends

### CPU — Cholesky (default)

Sparse direct CPU solve. Always available, no extra dependencies. Uses Apple
Accelerate on macOS when available, SuiteSparse CHOLMOD when available on other
platforms, and otherwise falls back to Eigen's simplicial factorizations.

```bash
vibestran --backend=cpu model.bdf
```

On macOS, the Accelerate sparse ordering can be selected with:

```bash
VIBESTRAN_ACCELERATE_ORDER=metis vibestran --backend=cpu model.bdf
```

### CPU — PCG

Eigen Preconditioned Conjugate Gradient with Incomplete Cholesky preconditioning.
Iterative solver with O(nnz) memory — suitable for very large systems where the
direct Cholesky fill-in would exhaust RAM. This is slower than the direct (Cholesky) solution.

```bash
vibestran --backend=cpu-pcg model.bdf
```

### CUDA — cuDSS

Uses [NVIDIA cuDSS](https://developer.nvidia.com/cudss) for GPU-resident sparse
direct solve. Requires CUDA toolkit ≥ 11 and cuDSS ≥ 0.7. Also used as the
shift-and-invert linear solver inside the IRL eigensolver for SOL 103.

**Solver strategy:**
1. Sparse Cholesky (`CUDSS_MTYPE_SPD`) — optimal for SPD FEM stiffness matrices
2. Sparse LU (`CUDSS_MTYPE_GENERAL`) fallback for non-SPD or ill-conditioned matrices

**Memory management:**
- After symbolic analysis, cuDSS memory estimates are logged (device/host stable and peak).
- If the estimated device peak exceeds 85% of free GPU memory, hybrid host/device memory
  mode is enabled automatically so factor storage spills to host RAM.
- If analysis itself fails with an allocation error, hybrid mode is retried before giving up.
- In theory Hybrid mode allows for very large systems to be solved, but in practice that hasn't been achieved (yet).

```bash
vibestran --backend=cuda model.bdf
```

The cuDSS direct backend also supports a float32 solve path for reduced VRAM
usage:

```bash
vibestran --backend=cuda --cuda-precision=fp32 model.bdf
```

#### Installing cuDSS (Ubuntu 24.04)

Download the deb from Nvidia, then install:
```bash
# Add the local repo and install
sudo dpkg -i cudss-local-repo-ubuntu2404-*.deb
sudo apt update
sudo apt install libcudss0-cuda-12 libcudss0-dev-cuda-12
```

### CUDA — PCG

GPU Preconditioned Conjugate Gradient with IC0 → ILU0 → Jacobi fallback preconditioning,
using cuSPARSE SpSV for triangular solves and cuBLAS for vector operations.
O(nnz) device memory — no factorization fill-in — making it suitable for systems too large
for cuDSS even with hybrid host/device memory mode. This is slower than the cuDSS direct solve for most cases.

```bash
vibestran --backend=cuda-pcg model.bdf
```

Requires CUDA toolkit ≥ 11 with cuBLAS and cuSPARSE (both included in the standard toolkit).

### Vulkan - Experimental

PCG iterative solver using Vulkan compute shaders. Requires Vulkan SDK, `glslc`, and `xxd`. Unfortunately performance is terrible right now, it's orders of magnitude slower than the base Eigen Cholesky CPU solver.

```bash
vibestran --backend=vulkan model.bdf
```

## Dependencies

| Dependency | Required | Purpose |
|---|---|---|
| C++20 compiler (GCC 11+, Clang 13+) | Yes | |
| Eigen 3.4+ | Yes | Core linear algebra and sparse CPU backends |
| Spectra | Yes | CPU generalized eigensolver for SOL 103 |
| TBB | No | Optional parallel execution support |
| Google Test | Yes (tests) | Fetched by Meson WrapDB |
| CUDA toolkit ≥ 11 + cuDSS ≥ 0.7 | No | CUDA backend |
| Vulkan SDK + glslc + xxd | No | Vulkan backend |
