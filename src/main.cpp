// src/main.cpp
// Command-line driver: nastran_solver [--backend=<cpu|vulkan|cuda>] <input.bdf> [output.f06]
//
// Default backend is the Eigen CPU solver.  Use --backend to select a GPU solver
// when available:
//   --backend=cpu               Eigen sparse Cholesky (always available, default)
//   --backend=vulkan            Vulkan PCG (requires Vulkan SDK; note: higher latency than CUDA)
//   --backend=cuda              NVIDIA cuSOLVER sparse Cholesky (requires CUDA toolkit)
//   --cuda-single-precision     Use float32 instead of float64 for the CUDA solve.
//                               Halves device memory usage; useful for very large problems
//                               that exceed GPU memory in double precision.

#include "io/bdf_parser.hpp"
#include "io/results.hpp"
#include "solver/linear_static.hpp"
#include "solver/solver_backend.hpp"
#ifdef HAVE_VULKAN
#  include "solver/vulkan_solver_backend.hpp"
#endif
#ifdef HAVE_CUDA
#  include "solver/cuda_solver_backend.hpp"
#endif
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string_view>

enum class BackendChoice { Auto, Cpu, Vulkan, Cuda };

static void print_usage() {
    std::cerr <<
        "Usage: nastran_solver [--backend=<cpu|vulkan|cuda>] [--cuda-single-precision] <input.bdf> [output.f06]\n"
        "  --backend=cpu              Eigen sparse Cholesky CPU solver (default)\n"
        "  --backend=vulkan           Vulkan PCG GPU solver (requires Vulkan)\n"
        "  --backend=cuda             CUDA cuSOLVER sparse Cholesky (requires CUDA)\n"
        "  --cuda-single-precision    Use float32 for CUDA solve (halves GPU memory usage)\n";
}

int main(int argc, char* argv[]) {
    // ── Argument parsing ─────────────────────────────────────────────────────
    BackendChoice backend_choice = BackendChoice::Auto;
    bool cuda_single_precision = false;
    int  positional = 0;
    std::filesystem::path bdf_path;
    std::filesystem::path f06_path;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--cuda-single-precision") {
            cuda_single_precision = true;
        } else if (arg.starts_with("--backend=")) {
            std::string_view val = arg.substr(std::string_view("--backend=").size());
            if (val == "cpu")         backend_choice = BackendChoice::Cpu;
            else if (val == "vulkan") backend_choice = BackendChoice::Vulkan;
            else if (val == "cuda")   backend_choice = BackendChoice::Cuda;
            else {
                std::cerr << "Unknown backend '" << val << "'. Valid: cpu, vulkan, cuda\n";
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
        nastran::Model model = nastran::BdfParser::parse_file(bdf_path);

        std::cout << "  Nodes:    " << model.nodes.size()             << "\n";
        std::cout << "  Elements: " << model.elements.size()          << "\n";
        std::cout << "  Materials:" << model.materials.size()         << "\n";
        std::cout << "  Subcases: " << model.analysis.subcases.size() << "\n";

        // ── Backend selection ─────────────────────────────────────────────────
        std::unique_ptr<nastran::SolverBackend> backend;

        if (backend_choice == BackendChoice::Cuda) {
#ifdef HAVE_CUDA
            auto cu = nastran::CudaSolverBackend::try_create(cuda_single_precision);
            if (cu.has_value()) {
                backend = std::make_unique<nastran::CudaSolverBackend>(std::move(*cu));
            } else {
                std::cerr << "CUDA backend requested but no CUDA device found\n";
                return 1;
            }
#else
            std::cerr << "CUDA backend was not compiled into this build\n";
            return 1;
#endif
        } else if (backend_choice == BackendChoice::Vulkan) {
#ifdef HAVE_VULKAN
            auto vk = nastran::VulkanSolverBackend::try_create();
            if (vk.has_value()) {
                backend = std::make_unique<nastran::VulkanSolverBackend>(std::move(*vk));
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
        // (Auto no longer attempts to use Vulkan or CUDA automatically — the Vulkan
        //  backend has higher dispatch latency that degrades throughput for many
        //  FEM problem sizes.  Users who want GPU acceleration should select it
        //  explicitly with --backend=cuda or --backend=vulkan.)

        if (!backend)
            backend = std::make_unique<nastran::EigenSolverBackend>();

        std::cout << "Solving with: " << backend->name() << "\n";
        nastran::LinearStaticSolver solver(std::move(backend));
        nastran::SolverResults results = solver.solve(model);

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Solution complete in " << elapsed << " s\n";

        nastran::F06Writer::write(results, model, f06_path);
        std::cout << "Results written: " << f06_path << "\n";

    } catch (const nastran::ParseError& e) {
        std::cerr << "Parse error: " << e.what() << "\n";
        return 2;
    } catch (const nastran::SolverError& e) {
        std::cerr << "Solver error: " << e.what() << "\n";
        return 3;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 4;
    }
    return 0;
}
