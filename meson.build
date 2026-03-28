# meson.build — Vibestran build system
# Requires: Meson >= 1.0, C++20 compiler, Eigen3
# Optional: Vulkan SDK (vulkan headers + loader) and glslc for GPU acceleration
# Usage:
#   meson setup build --buildtype=release
#   cd build && meson compile
#   meson test

project('vibestran', 'cpp',
  version : '0.1.0',
  default_options : [
    'cpp_std=c++20',
    'warning_level=3',
    'b_ndebug=if-release',
    'default_library=static',
  ]
)

fs     = import('fs')
python = import('python').find_installation('python3')
bin2h  = files('tools/bin2h.py')

# ── Compiler setup ────────────────────────────────────────────────────────────
cxx = meson.get_compiler('cpp')

# Compiler-specific flags
extra_args = []
if cxx.get_id() == 'gcc' or cxx.get_id() == 'clang'
  extra_args += ['-Wno-unused-parameter', '-Wfatal-errors']
  # Enable AVX2 for release builds (benefits Eigen)
  if get_option('buildtype') == 'release'
    extra_args += ['-march=native', '-ffast-math', '-fno-finite-math-only']
  endif
elif cxx.get_id() == 'msvc'
  extra_args += ['/wd4100', '/EHsc']
  if get_option('buildtype') == 'release'
    extra_args += ['/arch:AVX2', '/fp:fast']
  endif
endif

add_project_arguments(extra_args, language : 'cpp')

# ── Eigen dependency ──────────────────────────────────────────────────────────
# First try pkg-config, then fall back to third_party/ bundled headers
eigen_dep = dependency('eigen3',
  required     : false,
  fallback     : ['eigen3', 'eigen_dep'],
  include_type : 'system',   # use -isystem so GCC suppresses third-party warnings
)

if not eigen_dep.found()
  if fs.is_dir('third_party/eigen/Eigen')
    message('Using bundled Eigen from third_party/')
    eigen_dep = declare_dependency(include_directories : include_directories('third_party/eigen', is_system: true))
  else
    error('Eigen3 not found. Install libeigen3-dev (Linux), eigen (Homebrew), ' +
          'or run: meson subprojects download eigen3')
  endif
else
  message('Using system Eigen3')
endif

# ── spdlog structured logging ─────────────────────────────────────────────────
spdlog_dep = dependency('spdlog', required : true)

# ── Spectra eigensolver (header-only, bundled) ────────────────────────────────
spectra_dep = dependency('spectra', required : false)
if not spectra_dep.found()
  if fs.is_dir('third_party/spectra/include/Spectra')
    message('Using bundled Spectra from third_party/spectra/')
    spectra_dep = declare_dependency(
      include_directories : include_directories('third_party/spectra/include'))
  else
    error('Spectra not found. Download from https://github.com/yixuan/spectra ' +
          'and place headers in third_party/spectra/include/')
  endif
else
  message('Using system Spectra')
endif

# ── Apple Accelerate sparse solver backend (optional, macOS only) ────────────
# Accelerate.framework ships with every macOS install (10.13+) and provides
# direct sparse Cholesky via Eigen's AccelerateSupport header — no Homebrew
# packages required.  On macOS this replaces CHOLMOD as the fast direct solver.
accelerate_dep  = []
accelerate_args = []
have_accelerate = false
if host_machine.system() == 'darwin'
  _accel = dependency('appleframeworks', modules : ['Accelerate'], required : false)
  if _accel.found()
    accelerate_dep  = [_accel]
    accelerate_args = ['-DHAVE_ACCELERATE=1']
    have_accelerate = true
    message('Apple Accelerate sparse solver: enabled')
  else
    message('Apple Accelerate sparse solver: disabled')
  endif
endif

# ── SuiteSparse CHOLMOD backend (optional) ───────────────────────────────────
# When present, Eigen's CholmodDecomposition replaces SimplicialLLT, providing
# multithreaded supernodal Cholesky and better AMD+METIS reordering.
# Ubuntu's libsuitesparse-dev does not ship a cholmod.pc, so we use
# find_library() + manual include path rather than dependency().
_ss_opt      = get_option('use_suitesparse')
_ss_required = _ss_opt.enabled()
_cholmod_lib = cxx.find_library('cholmod', required : _ss_required)

# Build a list of candidate SuiteSparse header directories by platform
_ss_candidates = []
if host_machine.system() == 'linux'
  _ss_candidates = ['/usr/include/suitesparse', '/usr/local/include/suitesparse']
elif host_machine.system() == 'darwin'
  _ss_candidates = [
    '/opt/homebrew/include/suitesparse',   # Apple Silicon
    '/usr/local/include/suitesparse',      # Intel Mac
  ]
# On Windows, vcpkg/conda typically put suitesparse headers on the standard
# include path or a user-specified dir; skip explicit path and rely on the
# compiler's default search.
endif

_ss_inc_dir = ''
foreach c : _ss_candidates
  if fs.is_dir(c)
    _ss_inc_dir = c
    break
  endif
endforeach

_have_ss_inc = _ss_inc_dir != ''
_cholmod_inc = _have_ss_inc ? include_directories(_ss_inc_dir, is_system : true) : []

have_cholmod     = _cholmod_lib.found()
cholmod_args     = []
cholmod_lib_deps = []
cholmod_incdirs  = []
if have_cholmod
  cholmod_args     = ['-DEIGEN_CHOLMOD_SUPPORT']
  cholmod_lib_deps = [declare_dependency(
    dependencies        : _cholmod_lib,
    include_directories : _cholmod_inc,
  )]
  cholmod_incdirs = _have_ss_inc ? [_cholmod_inc] : []
  message('SuiteSparse CHOLMOD backend: enabled')
else
  message('SuiteSparse CHOLMOD backend: disabled (install libsuitesparse-dev to enable)')
endif

# ── Thread / parallel execution support ──────────────────────────────────────
thread_dep  = dependency('threads')
tbb_dep     = dependency('tbb', required : false)
if not tbb_dep.found()
  message('TBB not found; parallel execution handled by OpenMP and Eigen internals')
endif
openmp_dep  = dependency('openmp', required : false)
if not openmp_dep.found() and host_machine.system() == 'darwin'
  message('OpenMP not found. For parallelism with AppleClang: brew install libomp')
  message('Then reconfigure with -Dcpp_args=-I/opt/homebrew/opt/libomp/include etc.')
endif

# ── Vulkan compute backend (optional) ────────────────────────────────────────
vulkan_dep = dependency('vulkan', required : false)
glslc      = find_program('glslc', required : false)

vulkan_sources = []
vulkan_compile_args = []
spirv_hdrs = []
have_vulkan_backend = vulkan_dep.found() and glslc.found()

if have_vulkan_backend
  shader_names = ['spmv', 'dot_reduce', 'axpby', 'jacobi_precond',
                  'spmv_d', 'dot_reduce_d', 'axpby_d', 'jacobi_precond_d']
  foreach name : shader_names
    spv = custom_target(name + '_spv',
      input   : 'shaders' / name + '.comp',
      output  : name + '.spv',
      command : [glslc, '--target-env=vulkan1.1', '-O', '@INPUT@', '-o', '@OUTPUT@'],
    )
    hdr = custom_target(name + '_hdr',
      input   : spv,
      output  : name + '_spv.h',
      command : [python, bin2h, name + '_spv', '@INPUT@', '@OUTPUT@'],
    )
    spirv_hdrs += hdr
  endforeach

  vulkan_sources = files(
    'src/solver/vulkan_context.cpp',
    'src/solver/vulkan_buffer.cpp',
    'src/solver/vulkan_pipelines.cpp',
    'src/solver/vulkan_pcg_full.cpp',
    'src/solver/vulkan_pcg_tiled.cpp',
    'src/solver/vulkan_solver_backend.cpp',
  )
  vulkan_compile_args = ['-DHAVE_VULKAN=1']
  message('Vulkan compute backend: enabled')
else
  message('Vulkan compute backend: disabled (need vulkan and glslc)')
endif

# ── CUDA sparse solver backend (optional) ────────────────────────────────────
# Requires: CUDA toolkit >= 11.0 with cuDSS >= 0.7.
# The CUDA compiler (nvcc) is registered via add_languages(); when it is absent
# the entire block is skipped and the build proceeds without CUDA support.
have_cuda_compiler = add_languages('cuda', required: false, native: false)
cuda_sources       = []
cuda_compile_args  = []
cuda_lib_deps      = []
have_cuda_backend  = false

if have_cuda_compiler
  # nvcc's bundled host compiler (g++-12) supports C++20 except for <format>.
  # The .cu file uses only exceptions.hpp (no <format>) to stay within g++-12
  # compatibility.  Set --std=c++20 so other C++20 features (span, optional,
  # string_view) are available in host code.
  add_project_arguments('--std=c++20', language: 'cuda')

  # Suppress NVCC hd_warnings from Eigen headers: Eigen's __host__ __device__
  # math functions call constexpr __host__-only std:: functions. These are
  # harmless — Eigen guards the GPU paths correctly — but generate thousands of
  # lines of noise.  #pragma hd_warning_disable cannot appear at file scope or
  # before includes (NVCC requires it immediately before a function declaration),
  # so -diag-suppress is the only viable option for third-party headers.
  add_project_arguments('-diag-suppress=20011,20013,20014,20015', language: 'cuda')
  # GCC (host compiler) warns about Eigen's NVCC-specific #pragma diag_suppress
  # directives inside DisableStupidWarnings.h; suppress at the host level.
  add_project_arguments('-Xcompiler=-Wno-unknown-pragmas', language: 'cuda')

  # Eigen sets EIGEN_DEFAULT_ALIGN_BYTES based on the target architecture.
  # With -march=native on AVX2 hosts this is 32; without it, 16.  Both the
  # .cpp and .cu translation units instantiate Eigen templates (VectorXd,
  # SparseMatrix), so they MUST agree on alignment, otherwise Eigen objects
  # allocated on one side and freed on the other will use mismatched
  # allocators, causing double-free / heap corruption at runtime.
  # Forward the same -march=native flag to nvcc's host compiler.
  if get_option('buildtype') == 'release'
    add_project_arguments('-Xcompiler=-march=native', language: 'cuda')
  endif
endif

if have_cuda_compiler
  # Meson's 'dependency(cuda, modules:[...])' requires a canonical toolkit root
  # layout (/usr/local/cuda).  On systems where the CUDA toolkit is installed as
  # system packages (headers in /usr/include, libs in /usr/lib), the module-based
  # detection fails.  Use find_library() + declare_dependency() instead so we
  # work with both installation styles.
  #
  # cuDSS installs its library and headers into versioned subdirectories:
  #   headers: /usr/include/libcudss/12/
  #   library: /usr/lib/x86_64-linux-gnu/libcudss/12/
  # Build platform-specific cuDSS search paths
  _cudss_lib_dirs = []
  _cudss_inc_candidates = []
  if host_machine.system() == 'linux'
    _cudss_lib_dirs = [
      '/usr/lib/x86_64-linux-gnu/libcudss/12',
      '/usr/lib/aarch64-linux-gnu/libcudss/12',
      '/usr/local/cuda/lib64',
    ]
    _cudss_inc_candidates = ['/usr/include/libcudss/12', '/usr/local/cuda/include']
  elif host_machine.system() == 'windows'
    # CUDA_PATH is set by the NVIDIA installer; fall back to the conventional default
    _cuda_root = run_command('cmd', '/c', 'echo %CUDA_PATH%', check: false).stdout().strip()
    if _cuda_root == '' or _cuda_root == '%CUDA_PATH%'
      _cuda_root = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0'
    endif
    _cudss_lib_dirs = [_cuda_root / 'lib/x64']
    _cudss_inc_candidates = [_cuda_root / 'include']
  elif host_machine.system() == 'darwin'
    _cudss_lib_dirs = ['/usr/local/cuda/lib']
    _cudss_inc_candidates = ['/usr/local/cuda/include']
  endif

  _cudart_lib = cxx.find_library('cudart', required: false)
  _cudss_lib  = cxx.find_library('cudss', required: false, dirs: _cudss_lib_dirs)

  _cudss_inc_dir = ''
  foreach c : _cudss_inc_candidates
    if fs.is_dir(c)
      _cudss_inc_dir = c
      break
    endif
  endforeach
  _cudss_inc = _cudss_inc_dir != '' ? include_directories(_cudss_inc_dir) : []

  _cublas_lib   = cxx.find_library('cublas',   required: false)
  _cusparse_lib = cxx.find_library('cusparse', required: false)

  if _cudart_lib.found() and _cudss_lib.found()
    cuda_dep = declare_dependency(
      dependencies:        [_cudart_lib, _cudss_lib],
      include_directories: _cudss_inc,
    )
    cuda_sources      = files('src/solver/cuda_solver_backend.cu')
    cuda_compile_args = ['-DHAVE_CUDA=1']
    cuda_lib_deps     = [cuda_dep]
    have_cuda_backend = true
    message('CUDA sparse solver backend: enabled (cuDSS)')

    # PCG backend requires cuBLAS and cuSPARSE (standard CUDA toolkit libraries).
    if _cublas_lib.found() and _cusparse_lib.found()
      cuda_pcg_dep = declare_dependency(
        dependencies: [_cudart_lib, _cublas_lib, _cusparse_lib],
      )
      cuda_sources      += files(
        'src/solver/cuda_pcg_solver_backend.cu',
        'src/solver/cuda_mixed_pcg_solver_backend.cu',
      )
      cuda_lib_deps     += [cuda_pcg_dep]
      message('CUDA PCG solver backend: enabled (cuBLAS + cuSPARSE)')

      # Modal eigensolver requires cuDSS + cuBLAS + cuSPARSE together.
      cuda_eigensolver_dep = declare_dependency(
        dependencies: [_cudart_lib, _cudss_lib, _cublas_lib, _cusparse_lib],
        include_directories: _cudss_inc,
      )
      cuda_sources      += files('src/solver/cuda_eigensolver_backend.cu')
      cuda_compile_args += ['-DHAVE_CUDA_EIGENSOLVER=1']
      cuda_lib_deps     += [cuda_eigensolver_dep]
      message('CUDA eigensolver backend: enabled (cuDSS + cuBLAS + cuSPARSE)')
    else
      message('CUDA PCG solver backend: disabled (cublas or cusparse not found)')
      message('CUDA eigensolver backend: disabled (cublas or cusparse not found)')
    endif
  else
    message('CUDA sparse solver backend: disabled (cudart/cudss not found)')
  endif
else
  message('CUDA sparse solver backend: disabled (no CUDA compiler / nvcc not in PATH)')
endif

# ── cppcheck ─────────────────────────────────────────────────────────────────
cppcheck = find_program('cppcheck', required: false)
if cppcheck.found()
  cppcheck_suppressions = [
    '--suppress=*:' + join_paths(meson.project_source_root(), 'third_party') + '/*',
    '--suppress=*:' + join_paths(meson.project_source_root(), 'subprojects') + '/*',
    '--suppress=*:/opt/homebrew/*',
    '--suppress=*:/usr/local/*',
    '--suppress=*:/usr/include/*',
    '--suppress=*:/Library/Developer/CommandLineTools/*',
    '--suppress=*:/Applications/Xcode.app/*',
    '--suppress=*:*/lib/gcc/*',
  ]
  run_target('cppcheck',
    command: [
      cppcheck,
      '--project=' + join_paths(meson.project_build_root(), 'compile_commands.json'),
      '--enable=all',
      '--inline-suppr',
      '--quiet',
      '--suppress=missingIncludeSystem',
      '--suppress=checkersReport',
      '--suppress=normalCheckLevelMaxBranches',
      '--suppress=checkLevelNormal',
      '--suppress=unmatchedSuppression',
      cppcheck_suppressions,
      # Optional: Add extra arguments
      # '--force', 
      # '-j', '4',
    ]
  )
endif

# ── Source files ─────────────────────────────────────────────────────────────
solver_sources = files(
  # Vulkan sources are appended conditionally below
  # Core
  'src/core/logger.cpp',
  'src/core/model.cpp',
  'src/core/dof_map.cpp',
  'src/core/sparse_matrix.cpp',
  'src/core/coord_sys.cpp',
  'src/core/mpc_handler.cpp',
  # Elements
  'src/elements/cquad4.cpp',
  'src/elements/ctria3.cpp',
  'src/elements/line_elements.cpp',
  'src/elements/solid_elements.cpp',
  'src/elements/element_factory.cpp',
  'src/elements/rbe_constraints.cpp',
  # I/O
  'src/io/bdf_parser.cpp',
  'src/io/inp_parser.cpp',
  'src/io/results.cpp',
  'src/io/op2_writer.cpp',
  'src/io/csv_writer.cpp',
  # Solver
  'src/solver/solver_backend.cpp',
  'src/solver/analysis_support.cpp',
  'src/solver/linear_static.cpp',
  'src/solver/linear_static_pressure.cpp',
  'src/solver/eigensolver_backend.cpp',
  'src/solver/modal.cpp',
)

# Append Vulkan and CUDA sources when available
solver_sources += vulkan_sources
solver_sources += cuda_sources

# Include directories
inc     = include_directories('include')
src_inc = include_directories('src/solver') # for internal vulkan_pipelines.hpp

# ── Library (solver logic, no main) ──────────────────────────────────────────
tbb_args = tbb_dep.found() ? ['-DHAVE_TBB=1'] : []

solver_lib_deps = [eigen_dep, spectra_dep, spdlog_dep, thread_dep]
if tbb_dep.found()
  solver_lib_deps += tbb_dep
endif
if openmp_dep.found()
  solver_lib_deps += openmp_dep
endif
if have_vulkan_backend
  solver_lib_deps += vulkan_dep
endif
if have_cuda_backend
  solver_lib_deps += cuda_lib_deps
endif
if have_cholmod
  solver_lib_deps += cholmod_lib_deps
endif
if have_accelerate
  solver_lib_deps += accelerate_dep
endif

solver_lib = static_library(
  'vibestran',
  [solver_sources, spirv_hdrs],
  include_directories : [inc, src_inc],
  dependencies        : solver_lib_deps,
  cpp_args            : vulkan_compile_args + cuda_compile_args + cholmod_args + tbb_args + accelerate_args,
  install             : false,
)

solver_lib_dep = declare_dependency(
  link_with           : solver_lib,
  include_directories : [inc, src_inc],
  dependencies        : solver_lib_deps,
  compile_args        : vulkan_compile_args + cuda_compile_args + cholmod_args,
)

# ── Executable ───────────────────────────────────────────────────────────────
vibestran_exe = executable(
  'vibestran',
  'src/main.cpp',
  dependencies : solver_lib_dep,
  install      : true,
)

# ── Tests ────────────────────────────────────────────────────────────────────
# Google Test
gtest_dep = dependency('gtest',
  main         : true,
  fallback     : ['gtest', 'gtest_dep'],
  include_type : 'system',
)

# Unit tests
unit_test_sources = files(
  'tests/unit/test_dof_map.cpp',
  'tests/unit/test_bdf_parser.cpp',
  'tests/unit/test_eigensolver.cpp',
  'tests/unit/test_elements.cpp',
  'tests/unit/test_types.cpp',
  'tests/unit/test_sparse_matrix.cpp',
  'tests/unit/test_pcg_solver.cpp',
  'tests/unit/test_output_writers.cpp',
  'tests/unit/test_coord_sys.cpp',
  'tests/unit/test_mpc_handler.cpp',
  'tests/unit/test_inp_parser.cpp',
)

unit_tests = executable(
  'unit_tests',
  unit_test_sources,
  dependencies : [solver_lib_dep, gtest_dep],
  include_directories : inc,
)

test('unit_tests', unit_tests,
  timeout : 60,
  suite   : 'unit',
)

# Integration tests
integration_test_sources = files(
  'tests/integration/test_integration.cpp',
)

integration_tests = executable(
  'integration_tests',
  integration_test_sources,
  dependencies : [solver_lib_dep, gtest_dep],
  include_directories : inc,
)

test('integration_tests', integration_tests,
  timeout : 120,
  suite   : 'integration',
  # Integration tests are more numerically sensitive; run with verbose output
  args    : ['--gtest_output=xml:' + meson.project_build_root() + '/integration_test_results.xml'],
)

test('e2e_tests', python,
  timeout : 300,
  suite   : 'e2e',
  args    : [
    files('tools/e2e_test_runner.py'),
    '--solver', vibestran_exe.full_path(),
    '--bdf-dir', join_paths(meson.project_source_root(), 'tests', 'e2e', 'cases'),
    '--results-dir', join_paths(meson.project_build_root(), 'e2e_results'),
  ],
)

# ── Vulkan unit tests (built only when Vulkan backend is available) ────────────
if have_vulkan_backend
  vulkan_tests = executable(
    'vulkan_tests',
    files('tests/unit/test_vulkan_solver.cpp'),
    dependencies        : [solver_lib_dep, gtest_dep],
    include_directories : [inc, src_inc],
  )
  test('vulkan_tests', vulkan_tests,
    timeout : 120,
    suite   : 'unit',
  )
endif

# ── CUDA unit tests (built only when CUDA backend is available) ───────────────
if have_cuda_backend
  cuda_tests = executable(
    'cuda_tests',
    files('tests/unit/test_cuda_solver.cpp'),
    dependencies        : [solver_lib_dep, gtest_dep],
    include_directories : inc,
  )
  test('cuda_tests', cuda_tests,
    timeout : 120,
    suite   : 'unit',
  )
endif

# ── Summary ───────────────────────────────────────────────────────────────────
summary({
  'Build type'        : get_option('buildtype'),
  'C++ std'           : 'C++20',
  'Eigen'             : eigen_dep.found(),
  'Spectra'           : spectra_dep.found(),
  'CHOLMOD backend'   : have_cholmod,
  'Accelerate backend': have_accelerate,
  'Vulkan backend'    : have_vulkan_backend,
  'CUDA backend'      : have_cuda_backend,
  'Solver lib'        : solver_lib.name(),
  'Executable'        : vibestran_exe.name(),
}, section : 'Vibestran')
