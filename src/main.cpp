// src/main.cpp
// Command-line driver: vibetran [options] <input.bdf> [output.f06]
//
// Default backend is the Eigen CPU solver.  Use --backend to select a GPU
// solver when available:
//   --backend=cpu               Eigen sparse Cholesky (always available, default)
//   --backend=cpu-pcg           Eigen PCG + IncompleteCholesky (low memory)
//   --backend=cuda              NVIDIA cuDSS sparse direct solver (requires CUDA)
//                               For SOL 103 modal: uses the CUDA shift-invert
//                               Lanczos eigensolver (requires cuDSS+cuBLAS+cuSPARSE)
//   --backend=cuda-pcg          NVIDIA cuBLAS/cuSPARSE PCG (low memory, requires CUDA)
//                               For SOL 103 modal: uses the CUDA eigensolver
//                               (requires cuDSS+cuBLAS+cuSPARSE)
//   --backend=vulkan            Vulkan PCG (requires Vulkan SDK)
//                               SOL 103 modal is not supported with --backend=vulkan.
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
//
// Logging:
//   --log-file=<path>  Write all log output to <path> in addition to stdout.

#include "core/logger.hpp"
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
#ifdef HAVE_CUDA_EIGENSOLVER
#include "solver/cuda_eigensolver_backend.hpp"
#endif
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <memory>
#include <spdlog/spdlog.h>
#include <string_view>

enum class BackendChoice { Auto, Cpu, CpuPCG, Vulkan, Cuda, CudaPCG };

static void print_usage() {
  spdlog::error("Usage: vibetran "
                "[--backend=<cpu|cpu-pcg|vulkan|cuda|cuda-pcg>]\n"
                "                      [--cuda-single-precision] [--csv]\n"
                "                      [--log-file=<path>]\n"
                "                      <input.bdf|input.inp> [output.f06]\n"
                "  --backend=cpu              Eigen sparse Cholesky CPU solver "
                "(default)\n"
                "  --backend=cpu-pcg          Eigen PCG + IncompleteCholesky "
                "CPU solver (low memory)\n"
                "  --backend=vulkan           Vulkan PCG GPU solver (requires "
                "Vulkan; SOL 101 only)\n"
                "  --backend=cuda             CUDA cuDSS direct solver "
                "(requires CUDA; SOL 101 + SOL 103)\n"
                "  --backend=cuda-pcg         CUDA PCG + IC0/ILU0 GPU solver "
                "(requires CUDA; SOL 101 + SOL 103)\n"
                "  --cuda-single-precision    Use float32 for CUDA solve "
                "(halves GPU memory usage)\n"
                "  --csv                      Write CSV output even if "
                "PARAM,CSVOUT is not in the BDF\n"
                "  --log-file=<path>          Also write all log output to "
                "this file\n"
                "  OMP_NUM_THREADS=N          Limit CPU solver threads, e.g.:\n"
                "                             OMP_NUM_THREADS=8 vibetran "
                "input.bdf\n");
}

int main(int argc, char *argv[]) {
  // ── Argument parsing ─────────────────────────────────────────────────────
  BackendChoice backend_choice = BackendChoice::Auto;
  bool cuda_single_precision = false;
  bool force_csv = false;
  std::filesystem::path log_file_path;
  int positional = 0;
  std::filesystem::path bdf_path;
  std::filesystem::path f06_path;

  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "--cuda-single-precision") {
      cuda_single_precision = true;
    } else if (arg == "--csv") {
      force_csv = true;
    } else if (arg.starts_with("--log-file=")) {
      log_file_path = std::string(arg.substr(std::string_view("--log-file=").size()));
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
        // init_logger with no file so error() goes somewhere
        vibetran::init_logger();
        spdlog::error("Unknown backend '{}'. Valid: cpu, cpu-pcg, vulkan, cuda, cuda-pcg", val);
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
      vibetran::init_logger();
      spdlog::error("Unexpected argument: {}", arg);
      print_usage();
      return 1;
    }
  }

  if (positional == 0) {
    vibetran::init_logger();
    print_usage();
    return 1;
  }

  // Initialise logger (and optional file sink) before any logging.
  vibetran::init_logger(log_file_path);

  if (f06_path.empty())
    f06_path = std::filesystem::path(bdf_path).replace_extension(".f06");

  try {
    auto t0 = std::chrono::steady_clock::now();

    spdlog::info("Reading: {}", bdf_path.string());
    std::string ext = bdf_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    vibetran::Model model = (ext == ".inp")
        ? vibetran::InpParser::parse_file(bdf_path)
        : vibetran::BdfParser::parse_file(bdf_path);

    spdlog::info("  Nodes:    {}", model.nodes.size());
    spdlog::info("  Elements: {}", model.elements.size());
    spdlog::info("  Materials:{}", model.materials.size());
    spdlog::info("  Subcases: {}", model.analysis.subcases.size());

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
        spdlog::error("CUDA backend requested but no CUDA device found");
        return 1;
      }
#else
      spdlog::error("CUDA backend was not compiled into this build");
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
        spdlog::error("CUDA PCG backend requested but no CUDA device found");
        return 1;
      }
#else
      spdlog::error("CUDA PCG backend was not compiled into this build");
      return 1;
#endif
    } else if (backend_choice == BackendChoice::Vulkan) {
#ifdef HAVE_VULKAN
      auto vk = vibetran::VulkanSolverBackend::try_create();
      if (vk.has_value()) {
        backend =
            std::make_unique<vibetran::VulkanSolverBackend>(std::move(*vk));
      } else {
        spdlog::error("Vulkan backend requested but Vulkan is unavailable");
        return 1;
      }
#else
      spdlog::error("Vulkan backend was not compiled into this build");
      return 1;
#endif
    }
    // BackendChoice::Auto and BackendChoice::Cpu both default to Eigen CPU.
    if (!backend)
      backend = std::make_unique<vibetran::EigenSolverBackend>();

    auto op2_path = std::filesystem::path(f06_path).replace_extension(".op2");

    if (model.analysis.sol == vibetran::SolutionType::Modal) {
      // ── SOL 103 Modal Analysis ───────────────────────────────────────────
      // Select eigensolver backend.
      // CUDA (cuda / cuda-pcg): use the GPU shift-invert Lanczos eigensolver
      //   when the CUDA eigensolver was compiled in; error otherwise.
      // Vulkan: not supported for modal analysis.
      // CPU / Auto: use Spectra.
      std::unique_ptr<vibetran::EigensolverBackend> eig_backend;

      if (backend_choice == BackendChoice::Cuda ||
          backend_choice == BackendChoice::CudaPCG) {
#ifdef HAVE_CUDA_EIGENSOLVER
        auto cu_eig = vibetran::CudaEigensolverBackend::try_create();
        if (!cu_eig.has_value()) {
          spdlog::error("CUDA eigensolver requested but no CUDA device found");
          return 1;
        }
        eig_backend = std::make_unique<vibetran::CudaEigensolverBackend>(
            std::move(*cu_eig));
#else
        spdlog::error("CUDA eigensolver was not compiled into this build "
                      "(requires cuDSS + cuBLAS + cuSPARSE)");
        return 1;
#endif
      } else if (backend_choice == BackendChoice::Vulkan) {
        spdlog::error("--backend=vulkan is not supported for SOL 103 modal "
                      "analysis; use --backend=cuda or --backend=cpu");
        return 1;
      } else {
        eig_backend = std::make_unique<vibetran::SpectraEigensolverBackend>();
      }

      spdlog::info("Solving with: {}", eig_backend->name());
      vibetran::ModalSolver modal_solver(std::move(eig_backend));
      vibetran::ModalSolverResults modal_results = modal_solver.solve(model);

      auto t1 = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(t1 - t0).count();
      spdlog::info("Solution complete in {:.3f} s", elapsed);

      vibetran::F06Writer::write_modal(modal_results, model, f06_path);
      spdlog::info("F06 written: {}", f06_path.string());

      vibetran::Op2Writer::write_modal(modal_results, model, op2_path);
      spdlog::info("OP2 written: {}", op2_path.string());

    } else {
      // ── SOL 101 Linear Static ────────────────────────────────────────────
      spdlog::info("Solving with: {}", backend->name());
      vibetran::LinearStaticSolver solver(std::move(backend));
      vibetran::SolverResults results = solver.solve(model);

      auto t1 = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(t1 - t0).count();
      spdlog::info("Solution complete in {:.3f} s", elapsed);

      vibetran::F06Writer::write(results, model, f06_path);
      spdlog::info("F06 written: {}", f06_path.string());

      vibetran::Op2Writer::write(results, model, op2_path);
      spdlog::info("OP2 written: {}", op2_path.string());

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
        spdlog::info("CSV written: {}.node.csv / .elem.csv", csv_stem.string());
      }
    }

  } catch (const vibetran::ParseError &e) {
    spdlog::error("Parse error: {}", e.what());
    return 2;
  } catch (const vibetran::SolverError &e) {
    spdlog::error("Solver error: {}", e.what());
    return 3;
  } catch (const std::exception &e) {
    spdlog::error("Error: {}", e.what());
    return 4;
  }
  return 0;
}
