// src/main.cpp
// Command-line driver: vibestran [options] <input.bdf> [output.f06]
//
// Default backend is the Eigen CPU solver.  Use --backend to select a GPU
// solver when available:
//   --backend=cpu               Eigen sparse Cholesky (always available,
//   default)
//   --backend=cpu-pcg           Eigen PCG + IncompleteCholesky (low memory)
//   --backend=cuda              NVIDIA cuDSS sparse direct solver (requires
//   CUDA)
//                               For SOL 103 modal: uses the CUDA shift-invert
//                               Lanczos eigensolver (requires
//                               cuDSS+cuBLAS+cuSPARSE)
//   --backend=cuda-pcg          NVIDIA cuBLAS/cuSPARSE PCG (stable float64
//   path,
//                               low memory, requires CUDA)
//   --backend=cuda-pcg-mixed    NVIDIA mixed-precision PCG experiment
//                               SOL 103 modal is not supported with
//                               --backend=cuda-pcg-mixed.
//   --backend=vulkan            Vulkan PCG (requires Vulkan SDK)
//                               SOL 103 modal is not supported with
//                               --backend=vulkan.
//   --cuda-precision=<fp32|fp64> Precision for the CUDA cuDSS direct solver.
//
// Thread control (CPU backends):
//   Set OMP_NUM_THREADS before running to limit parallelism, e.g.:
//     OMP_NUM_THREADS=8 vibestran input.bdf
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
//
// Quality checks:
//   --checkmode=lenient  Override the deck and run quality checks in LENIENT
//                        mode for this invocation.

#include "core/logger.hpp"
#include "core/quality_checks.hpp"
#include "io/bdf_parser.hpp"
#include "io/inp_parser.hpp"
#include "io/results.hpp"
#include "solver/eigensolver_backend.hpp"
#include "solver/linear_static.hpp"
#include "solver/modal.hpp"
#include "solver/solver_backend.hpp"
#include <Eigen/Core>
#ifdef HAVE_VULKAN
#include "solver/vulkan_solver_backend.hpp"
#endif
#ifdef HAVE_CUDA
#include "solver/cuda_mixed_pcg_solver_backend.hpp"
#include "solver/cuda_pcg_solver_backend.hpp"
#include "solver/cuda_solver_backend.hpp"
#endif
#ifdef HAVE_CUDA_EIGENSOLVER
#include "solver/cuda_eigensolver_backend.hpp"
#endif
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <spdlog/spdlog.h>
#include <string_view>

enum class BackendChoice {
  Auto,
  Cpu,
  CpuPCG,
  Vulkan,
  Cuda,
  CudaPCG,
  CudaPCGMixed
};
enum class CudaPrecisionChoice { Float64, Float32 };

static void configure_eigen_threads() {
  int eigen_threads = Eigen::nbThreads();
  const char *omp_threads_env = std::getenv("OMP_NUM_THREADS");
  if (omp_threads_env != nullptr && omp_threads_env[0] != '\0') {
    char *end = nullptr;
    long parsed = std::strtol(omp_threads_env, &end, 10);
    if (end != omp_threads_env && *end == '\0' && parsed > 0 &&
        parsed <= std::numeric_limits<int>::max()) {
      eigen_threads = static_cast<int>(parsed);
    } else {
      spdlog::warn("Ignoring invalid OMP_NUM_THREADS='{}'", omp_threads_env);
    }
  }

  Eigen::setNbThreads(eigen_threads);
  if (omp_threads_env != nullptr && omp_threads_env[0] != '\0') {
    spdlog::info("Eigen CPU threads: {} (from OMP_NUM_THREADS={})",
                 Eigen::nbThreads(), omp_threads_env);
  } else {
    spdlog::info("Eigen CPU threads: {} (OpenMP default)", Eigen::nbThreads());
  }
}

static void print_usage() {
  spdlog::error(
      "Usage: vibestran "
      "[--backend=<cpu|cpu-pcg|vulkan|cuda|cuda-pcg|cuda-pcg-mixed>]\n"
      "                      [--cuda-precision=<fp32|fp64>] [--csv]\n"
      "                      [--log-file=<path>] "
      "[--checkmode=lenient]\n"
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
      "(stable float64 path; requires CUDA; SOL 101 + SOL 103)\n"
      "  --backend=cuda-pcg-mixed   CUDA mixed-precision PCG + "
      "iterative refinement experiment (requires CUDA; SOL 101 only)\n"
      "  --cuda-precision=fp32|fp64 Precision for --backend=cuda "
      "(default: fp64)\n"
      "  --csv                      Write CSV output even if "
      "PARAM,CSVOUT is not in the BDF\n"
      "  --log-file=<path>          Also write all log output to "
      "this file\n"
      "  --checkmode=lenient        Override the deck and run quality "
      "checks in LENIENT mode\n"
      "  OMP_NUM_THREADS=N          Limit CPU solver threads, e.g.:\n"
      "                             OMP_NUM_THREADS=8 vibestran "
      "input.bdf\n");
}

int main(int argc, const char *argv[]) {
  // ── Argument parsing ─────────────────────────────────────────────────────
  BackendChoice backend_choice = BackendChoice::Auto;
  CudaPrecisionChoice cuda_precision = CudaPrecisionChoice::Float64;
  bool force_csv = false;
  bool force_checkmode_lenient = false;
  std::filesystem::path log_file_path;
  int positional = 0;
  std::filesystem::path bdf_path;
  std::filesystem::path f06_path;

  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg.starts_with("--cuda-precision=")) {
      std::string_view val =
          arg.substr(std::string_view("--cuda-precision=").size());
      if (val == "fp32")
        cuda_precision = CudaPrecisionChoice::Float32;
      else if (val == "fp64")
        cuda_precision = CudaPrecisionChoice::Float64;
      else {
        vibestran::init_logger();
        spdlog::error("Unknown CUDA precision '{}'. Valid: fp32, fp64", val);
        print_usage();
        return 1;
      }
    } else if (arg == "--csv") {
      force_csv = true;
    } else if (arg.starts_with("--checkmode=")) {
      std::string_view val = arg.substr(std::string_view("--checkmode=").size());
      if (val == "lenient") {
        force_checkmode_lenient = true;
      } else {
        vibestran::init_logger();
        spdlog::error("Unknown checkmode '{}'. Valid: lenient", val);
        print_usage();
        return 1;
      }
    } else if (arg.starts_with("--log-file=")) {
      log_file_path =
          std::string(arg.substr(std::string_view("--log-file=").size()));
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
      else if (val == "cuda-pcg-mixed")
        backend_choice = BackendChoice::CudaPCGMixed;
      else {
        // init_logger with no file so error() goes somewhere
        vibestran::init_logger();
        spdlog::error("Unknown backend '{}'. Valid: cpu, cpu-pcg, vulkan, "
                      "cuda, cuda-pcg, cuda-pcg-mixed",
                      val);
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
      vibestran::init_logger();
      spdlog::error("Unexpected argument: {}", arg);
      print_usage();
      return 1;
    }
  }

  if (positional == 0) {
    vibestran::init_logger();
    print_usage();
    return 1;
  }

  // Initialise logger (and optional file sink) before any logging.
  vibestran::init_logger(log_file_path);
  configure_eigen_threads();

  if (cuda_precision != CudaPrecisionChoice::Float64 &&
      backend_choice != BackendChoice::Cuda) {
    spdlog::error("--cuda-precision is only supported with --backend=cuda");
    return 1;
  }

  if (f06_path.empty())
    f06_path = std::filesystem::path(bdf_path).replace_extension(".f06");

  try {
    auto t0 = std::chrono::steady_clock::now();

    spdlog::info("Reading: {}", bdf_path.string());
    std::string ext = bdf_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    vibestran::Model model = (ext == ".inp")
                                 ? vibestran::InpParser::parse_file(bdf_path)
                                 : vibestran::BdfParser::parse_file(bdf_path);

    if (force_checkmode_lenient) {
      model.params["CHECKMODE"] = "LENIENT";
      spdlog::info("CLI override: PARAM,CHECKMODE,LENIENT");
    }

    spdlog::info("  Nodes:    {}", model.nodes.size());
    spdlog::info("  Elements: {}", model.elements.size());
    spdlog::info("  Materials:{}", model.materials.size());
    spdlog::info("  Subcases: {}", model.analysis.subcases.size());

    // ── Pre-flight quality checks ─────────────────────────────────────────
    {
        auto tq0 = std::chrono::steady_clock::now();
        vibestran::QualityThresholds qt = vibestran::build_thresholds(model);
        vibestran::run_quality_checks(model, qt);
        auto tq1 = std::chrono::steady_clock::now();
        spdlog::info("quality checks: {:.3f} ms",
            std::chrono::duration<double, std::milli>(tq1 - tq0).count());
    }

    // ── Backend selection ─────────────────────────────────────────────────
    std::unique_ptr<vibestran::SolverBackend> backend;

    if (backend_choice == BackendChoice::CpuPCG) {
      backend = std::make_unique<vibestran::EigenPCGSolverBackend>();
    } else if (backend_choice == BackendChoice::Cuda) {
#ifdef HAVE_CUDA
      const auto precision = cuda_precision == CudaPrecisionChoice::Float32
                                 ? vibestran::CudaSolverPrecision::Float32
                                 : vibestran::CudaSolverPrecision::Float64;
      auto cu = vibestran::CudaSolverBackend::try_create(precision);
      if (cu.has_value()) {
        backend =
            std::make_unique<vibestran::CudaSolverBackend>(std::move(*cu));
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
      auto cu = vibestran::CudaPCGSolverBackend::try_create();
      if (cu.has_value()) {
        backend =
            std::make_unique<vibestran::CudaPCGSolverBackend>(std::move(*cu));
      } else {
        spdlog::error("CUDA PCG backend requested but no CUDA device found");
        return 1;
      }
#else
      spdlog::error("CUDA PCG backend was not compiled into this build");
      return 1;
#endif
    } else if (backend_choice == BackendChoice::CudaPCGMixed) {
#ifdef HAVE_CUDA
      auto cu = vibestran::CudaMixedPCGSolverBackend::try_create();
      if (cu.has_value()) {
        backend = std::make_unique<vibestran::CudaMixedPCGSolverBackend>(
            std::move(*cu));
      } else {
        spdlog::error(
            "CUDA mixed PCG backend requested but no CUDA device found");
        return 1;
      }
#else
      spdlog::error("CUDA mixed PCG backend was not compiled into this build");
      return 1;
#endif
    } else if (backend_choice == BackendChoice::Vulkan) {
#ifdef HAVE_VULKAN
      auto vk = vibestran::VulkanSolverBackend::try_create();
      if (vk.has_value()) {
        backend =
            std::make_unique<vibestran::VulkanSolverBackend>(std::move(*vk));
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
      backend = std::make_unique<vibestran::EigenSolverBackend>();

    auto op2_path = std::filesystem::path(f06_path).replace_extension(".op2");

    if (model.analysis.sol == vibestran::SolutionType::Modal) {
      // ── SOL 103 Modal Analysis ───────────────────────────────────────────
      // Select eigensolver backend.
      // CUDA (cuda / cuda-pcg): use the GPU shift-invert Lanczos eigensolver
      //   when the CUDA eigensolver was compiled in; error otherwise.
      // Vulkan: not supported for modal analysis.
      // CPU / Auto: use Spectra.
      std::unique_ptr<vibestran::EigensolverBackend> eig_backend;

      if (backend_choice == BackendChoice::Cuda ||
          backend_choice == BackendChoice::CudaPCG) {
#ifdef HAVE_CUDA_EIGENSOLVER
        auto cu_eig = vibestran::CudaEigensolverBackend::try_create();
        if (!cu_eig.has_value()) {
          spdlog::error("CUDA eigensolver requested but no CUDA device found");
          return 1;
        }
        eig_backend = std::make_unique<vibestran::CudaEigensolverBackend>(
            std::move(*cu_eig));
#else
        spdlog::error("CUDA eigensolver was not compiled into this build "
                      "(requires cuDSS + cuBLAS + cuSPARSE)");
        return 1;
#endif
      } else if (backend_choice == BackendChoice::CudaPCGMixed) {
        spdlog::error(
            "--backend=cuda-pcg-mixed is not supported for SOL 103 "
            "modal analysis; use --backend=cuda or --backend=cuda-pcg");
        return 1;
      } else if (backend_choice == BackendChoice::Vulkan) {
        spdlog::error("--backend=vulkan is not supported for SOL 103 modal "
                      "analysis; use --backend=cuda or --backend=cpu");
        return 1;
      } else {
        eig_backend = std::make_unique<vibestran::SpectraEigensolverBackend>();
      }

      spdlog::info("Solving with: {}", eig_backend->name());
      vibestran::ModalSolver modal_solver(std::move(eig_backend));
      vibestran::ModalSolverResults modal_results = modal_solver.solve(model);

      auto t1 = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(t1 - t0).count();
      spdlog::info("Solution complete in {:.3f} s", elapsed);

      vibestran::F06Writer::write_modal(modal_results, model, f06_path);
      spdlog::info("F06 written: {}", f06_path.string());

      vibestran::Op2Writer::write_modal(modal_results, model, op2_path);
      spdlog::info("OP2 written: {}", op2_path.string());

    } else {
      // ── SOL 101 Linear Static ────────────────────────────────────────────
      spdlog::info("Solving with: {}", backend->name());
      vibestran::LinearStaticSolver solver(std::move(backend));
      vibestran::SolverResults results = solver.solve(model);

      auto t1 = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(t1 - t0).count();
      spdlog::info("Solution complete in {:.3f} s", elapsed);

      vibestran::F06Writer::write(results, model, f06_path);
      spdlog::info("F06 written: {}", f06_path.string());

      vibestran::Op2Writer::write(results, model, op2_path);
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
        vibestran::CsvWriter::write(results, model, csv_stem);
        spdlog::info("CSV written: {}.node.csv / .elem.csv", csv_stem.string());
      }
    }

  } catch (const vibestran::ParseError &e) {
    spdlog::error("Parse error: {}", e.what());
    return 2;
  } catch (const vibestran::SolverError &e) {
    spdlog::error("Solver error: {}", e.what());
    return 3;
  } catch (const std::exception &e) {
    spdlog::error("Error: {}", e.what());
    return 4;
  }
  return 0;
}
