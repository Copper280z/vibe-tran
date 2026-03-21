// src/main.cpp
// Command-line driver: vibetran [options] <input.bdf> [output.f06]
//
// Default backend is the Eigen CPU solver.  Use --backend to select a GPU
// solver when available:
//   --backend=cpu               Eigen sparse Cholesky (always available, default)
//   --backend=cpu-pcg           Eigen PCG + IncompleteCholesky (low memory)
//   --backend=cuda              NVIDIA cuSOLVER sparse Cholesky (requires CUDA)
//   --backend=cuda-pcg          NVIDIA cuBLAS/cuSPARSE PCG (low memory, requires CUDA)
//   --backend=vulkan            Vulkan PCG (requires Vulkan SDK)
//   --cuda-single-precision     Use float32 instead of float64 for the CUDA solve.
//
// Thread control (CPU backends):
//   Set OMP_NUM_THREADS before running to limit parallelism, e.g.:
//     OMP_NUM_THREADS=8 vibetran input.bdf
//   On hyperthreaded CPUs, setting this to the physical core count often
//   improves performance.
//
// Output:
//   <stem>.f06   F06 text results (always written)
//   <stem>.op2   OP2 binary results (always written)
//   <stem>.node.csv  Nodal results CSV  \  written when PARAM,CSVOUT,YES is in
//   <stem>.elem.csv  Element results CSV/  the BDF, or when --csv is passed.

#include "io/bdf_parser.hpp"
#include "io/inp_parser.hpp"
#include "io/results.hpp"
#include "solver/eigensolver_backend.hpp"
#include "solver/linear_static.hpp"
#include "solver/modal.hpp"
#include "solver/solver_backend.hpp"
#ifdef HAVE_VULKAN
#include "solver/vulkan_solver_backend.hpp"
#endif
#ifdef HAVE_CUDA
#include "solver/cuda_pcg_solver_backend.hpp"
#include "solver/cuda_solver_backend.hpp"
#endif
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string_view>

enum class BackendChoice { Auto, Cpu, CpuPCG, Vulkan, Cuda, CudaPCG };

static void print_usage() {
  std::cerr << "Usage: vibetran "
               "[--backend=<cpu|cpu-pcg|vulkan|cuda|cuda-pcg>]\n"
               "                      [--cuda-single-precision] [--csv]\n"
               "                      <input.bdf|input.inp> [output.f06]\n"
               "  --backend=cpu              Eigen sparse Cholesky CPU solver "
               "(default)\n"
               "  --backend=cpu-pcg          Eigen PCG + IncompleteCholesky "
               "CPU solver (low memory)\n"
               "  --backend=vulkan           Vulkan PCG GPU solver (requires "
               "Vulkan)\n"
               "  --backend=cuda             CUDA cuDSS sparse direct solver "
               "(requires CUDA)\n"
               "  --backend=cuda-pcg         CUDA PCG + IC0/ILU0 GPU solver "
               "(low memory, requires CUDA)\n"
               "  --cuda-single-precision    Use float32 for CUDA solve "
               "(halves GPU memory usage)\n"
               "  --csv                      Write CSV output even if "
               "PARAM,CSVOUT is not in the BDF\n"
               "  OMP_NUM_THREADS=N          Limit CPU solver threads, e.g.:\n"
               "                             OMP_NUM_THREADS=8 vibetran "
               "input.bdf\n";
}

int main(int argc, char *argv[]) {
  // ── Argument parsing ─────────────────────────────────────────────────────
  BackendChoice backend_choice = BackendChoice::Auto;
  bool cuda_single_precision = false;
  bool force_csv = false;
  int positional = 0;
  std::filesystem::path bdf_path;
  std::filesystem::path f06_path;

  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "--cuda-single-precision") {
      cuda_single_precision = true;
    } else if (arg == "--csv") {
      force_csv = true;
    } else if (arg.starts_with("--backend=")) {
      std::string_view val = arg.substr(std::string_view("--backend=").size());
      if (val == "cpu")
        backend_choice = BackendChoice::Cpu;
      else if (val == "cpu-pcg")
        backend_choice = BackendChoice::CpuPCG;
      else if (val == "vulkan")
        backend_choice = BackendChoice::Vulkan;
      else if (val == "cuda")
        backend_choice = BackendChoice::Cuda;
      else if (val == "cuda-pcg")
        backend_choice = BackendChoice::CudaPCG;
      else {
        std::cerr << "Unknown backend '" << val
                  << "'. Valid: cpu, cpu-pcg, vulkan, cuda, cuda-pcg\n";
        print_usage();
        return 1;
      }
    } else if (positional == 0) {
      bdf_path = arg;
      ++positional;
    } else if (positional == 1) {
      f06_path = arg;
      ++positional;
    } else {
      std::cerr << "Unexpected argument: " << arg << "\n";
      print_usage();
      return 1;
    }
  }

  if (positional == 0) {
    print_usage();
    return 1;
  }

  if (f06_path.empty())
    f06_path = std::filesystem::path(bdf_path).replace_extension(".f06");

  try {
    auto t0 = std::chrono::steady_clock::now();

    std::cout << "Reading: " << bdf_path << "\n";
    std::string ext = bdf_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    vibetran::Model model = (ext == ".inp")
        ? vibetran::InpParser::parse_file(bdf_path)
        : vibetran::BdfParser::parse_file(bdf_path);

    std::cout << "  Nodes:    " << model.nodes.size() << "\n";
    std::cout << "  Elements: " << model.elements.size() << "\n";
    std::cout << "  Materials:" << model.materials.size() << "\n";
    std::cout << "  Subcases: " << model.analysis.subcases.size() << "\n";

    // ── Backend selection ─────────────────────────────────────────────────
    std::unique_ptr<vibetran::SolverBackend> backend;

    if (backend_choice == BackendChoice::CpuPCG) {
      backend = std::make_unique<vibetran::EigenPCGSolverBackend>();
    } else if (backend_choice == BackendChoice::Cuda) {
#ifdef HAVE_CUDA
      auto cu = vibetran::CudaSolverBackend::try_create(cuda_single_precision);
      if (cu.has_value()) {
        backend = std::make_unique<vibetran::CudaSolverBackend>(std::move(*cu));
      } else {
        std::cerr << "CUDA backend requested but no CUDA device found\n";
        return 1;
      }
#else
      std::cerr << "CUDA backend was not compiled into this build\n";
      return 1;
#endif
    } else if (backend_choice == BackendChoice::CudaPCG) {
#ifdef HAVE_CUDA
      auto cu =
          vibetran::CudaPCGSolverBackend::try_create(cuda_single_precision);
      if (cu.has_value()) {
        backend =
            std::make_unique<vibetran::CudaPCGSolverBackend>(std::move(*cu));
      } else {
        std::cerr << "CUDA PCG backend requested but no CUDA device found\n";
        return 1;
      }
#else
      std::cerr << "CUDA PCG backend was not compiled into this build\n";
      return 1;
#endif
    } else if (backend_choice == BackendChoice::Vulkan) {
#ifdef HAVE_VULKAN
      auto vk = vibetran::VulkanSolverBackend::try_create();
      if (vk.has_value()) {
        backend =
            std::make_unique<vibetran::VulkanSolverBackend>(std::move(*vk));
      } else {
        std::cerr << "Vulkan backend requested but Vulkan is unavailable\n";
        return 1;
      }
#else
      std::cerr << "Vulkan backend was not compiled into this build\n";
      return 1;
#endif
    }
    // BackendChoice::Auto and BackendChoice::Cpu both default to Eigen CPU.
    if (!backend)
      backend = std::make_unique<vibetran::EigenSolverBackend>();

    auto op2_path = std::filesystem::path(f06_path).replace_extension(".op2");

    if (model.analysis.sol == vibetran::SolutionType::Modal) {
      // ── SOL 103 Modal Analysis ───────────────────────────────────────────
      auto eig_backend =
          std::make_unique<vibetran::SpectraEigensolverBackend>();
      std::cout << "Solving with: " << eig_backend->name() << "\n";
      vibetran::ModalSolver modal_solver(std::move(eig_backend));
      vibetran::ModalSolverResults modal_results = modal_solver.solve(model);

      auto t1 = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(t1 - t0).count();
      std::cout << "Solution complete in " << elapsed << " s\n";

      vibetran::F06Writer::write_modal(modal_results, model, f06_path);
      std::cout << "F06 written: " << f06_path << "\n";

      vibetran::Op2Writer::write_modal(modal_results, model, op2_path);
      std::cout << "OP2 written: " << op2_path << "\n";

    } else {
      // ── SOL 101 Linear Static ────────────────────────────────────────────
      std::cout << "Solving with: " << backend->name() << "\n";
      vibetran::LinearStaticSolver solver(std::move(backend));
      vibetran::SolverResults results = solver.solve(model);

      auto t1 = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(t1 - t0).count();
      std::cout << "Solution complete in " << elapsed << " s\n";

      vibetran::F06Writer::write(results, model, f06_path);
      std::cout << "F06 written: " << f06_path << "\n";

      vibetran::Op2Writer::write(results, model, op2_path);
      std::cout << "OP2 written: " << op2_path << "\n";

      // ── Write CSV (if requested via --csv or PARAM,CSVOUT,YES) ──────────
      bool write_csv = force_csv;
      if (!write_csv) {
        auto it = model.params.find("CSVOUT");
        if (it != model.params.end() && it->second == "YES")
          write_csv = true;
      }
      if (write_csv) {
        auto csv_stem = std::filesystem::path(f06_path).replace_extension("");
        vibetran::CsvWriter::write(results, model, csv_stem);
        std::cout << "CSV written: " << csv_stem.string()
                  << ".node.csv / .elem.csv\n";
      }
    }

  } catch (const vibetran::ParseError &e) {
    std::cerr << "Parse error: " << e.what() << "\n";
    return 2;
  } catch (const vibetran::SolverError &e) {
    std::cerr << "Solver error: " << e.what() << "\n";
    return 3;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 4;
  }
  return 0;
}
