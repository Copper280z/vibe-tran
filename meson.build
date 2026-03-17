# meson.build — NastranSolver build system
# Requires: Meson >= 1.0, C++20 compiler, Eigen3
# Optional: Vulkan SDK (vulkan headers + loader) and glslc for GPU acceleration
# Usage:
#   meson setup build --buildtype=release
#   cd build && meson compile
#   meson test

project('nastran_solver', 'cpp',
  version : '0.1.0',
  default_options : [
    'cpp_std=c++20',
    'warning_level=3',
    'b_ndebug=if-release',
    'default_library=static',
  ]
)

fs = import('fs')

# ── Compiler setup ────────────────────────────────────────────────────────────
cxx = meson.get_compiler('cpp')

# Compiler-specific flags
extra_args = []
if cxx.get_id() == 'gcc' or cxx.get_id() == 'clang'
  extra_args += ['-Wno-unused-parameter', '-Wfatal-errors']
  # Enable AVX2 for release builds (benefits Eigen)
  if get_option('buildtype') == 'release'
    extra_args += ['-march=native', '-ffast-math']
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
  required : false,
  fallback : ['eigen3', 'eigen_dep'],
)

if not eigen_dep.found()
  # NOT CURRENTLY IMPLEMENTED
  # Use bundled Eigen headers in third_party/eigen
  # eigen_inc = include_directories('third_party/eigen')
  # eigen_dep = declare_dependency(include_directories : eigen_inc)
  message('Using bundled Eigen from third_party/')
  error('Bundled Eigen not currently implemented')
else
  message('Using system Eigen3')
endif

# ── Thread / parallel execution support ──────────────────────────────────────
thread_dep = dependency('threads')
tbb_dep    = dependency('tbb')

# ── Vulkan compute backend (optional) ────────────────────────────────────────
vulkan_dep = dependency('vulkan', required : false)
glslc      = find_program('glslc', required : false)
xxd        = find_program('xxd',   required : false)

vulkan_sources = []
vulkan_compile_args = []
spirv_hdrs = []
have_vulkan_backend = vulkan_dep.found() and glslc.found() and xxd.found()

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
      command : [xxd, '-i', '-n', name + '_spv', '@INPUT@', '@OUTPUT@'],
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
  message('Vulkan compute backend: disabled (need vulkan, glslc, and xxd)')
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
  _cudart_lib = cxx.find_library('cudart', required: false)
  _cudss_lib  = cxx.find_library('cudss',
    required: false,
    dirs: ['/usr/lib/x86_64-linux-gnu/libcudss/12'],
  )
  _cudss_inc  = include_directories('/usr/include/libcudss/12')

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
  else
    message('CUDA sparse solver backend: disabled (cudart/cudss not found)')
  endif
else
  message('CUDA sparse solver backend: disabled (no CUDA compiler / nvcc not in PATH)')
endif

# ── cppcheck ─────────────────────────────────────────────────────────────────
cppcheck = find_program('cppcheck', required: false)
if cppcheck.found()
  run_target('cppcheck',
    command: [
      cppcheck,
      '--project=' + join_paths(meson.project_build_root(), 'compile_commands.json'),
      '--enable=all',
      '--inline-suppr',
      '--quiet',
      '--suppress=missingIncludeSystem',
      '--suppress=checkersReport',
      '--quiet',
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
  'src/core/model.cpp',
  'src/core/dof_map.cpp',
  'src/core/sparse_matrix.cpp',
  # Elements
  'src/elements/cquad4.cpp',
  'src/elements/ctria3.cpp',
  'src/elements/solid_elements.cpp',
  'src/elements/element_factory.cpp',
  # I/O
  'src/io/bdf_parser.cpp',
  'src/io/results.cpp',
  # Solver
  'src/solver/solver_backend.cpp',
  'src/solver/linear_static.cpp',
)

# Append Vulkan and CUDA sources when available
solver_sources += vulkan_sources
solver_sources += cuda_sources

# Include directories
inc     = include_directories('include')
src_inc = include_directories('src/solver') # for internal vulkan_pipelines.hpp

# ── Library (solver logic, no main) ──────────────────────────────────────────
solver_lib_deps = [eigen_dep, thread_dep, tbb_dep]
if have_vulkan_backend
  solver_lib_deps += vulkan_dep
endif
if have_cuda_backend
  solver_lib_deps += cuda_lib_deps
endif

solver_lib = static_library(
  'nastransolver',
  [solver_sources, spirv_hdrs],
  include_directories : [inc, src_inc],
  dependencies        : solver_lib_deps,
  cpp_args            : vulkan_compile_args + cuda_compile_args,
  install             : false,
)

solver_lib_dep = declare_dependency(
  link_with           : solver_lib,
  include_directories : [inc, src_inc],
  dependencies        : solver_lib_deps,
  compile_args        : vulkan_compile_args + cuda_compile_args,
)

# ── Executable ───────────────────────────────────────────────────────────────
nastran_exe = executable(
  'nastran_solver',
  'src/main.cpp',
  dependencies : solver_lib_dep,
  install      : true,
)

# ── Tests ────────────────────────────────────────────────────────────────────
# Google Test
gtest_dep = dependency('gtest', main : true, fallback : ['gtest', 'gtest_dep'])

# Unit tests
unit_test_sources = files(
  'tests/unit/test_dof_map.cpp',
  'tests/unit/test_bdf_parser.cpp',
  'tests/unit/test_elements.cpp',
  'tests/unit/test_types.cpp',
  'tests/unit/test_sparse_matrix.cpp',
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
  'Build type'     : get_option('buildtype'),
  'C++ std'        : 'C++20',
  'Eigen'          : eigen_dep.found(),
  'Vulkan backend' : have_vulkan_backend,
  'CUDA backend'   : have_cuda_backend,
  'Solver lib'     : solver_lib.name(),
  'Executable'     : nastran_exe.name(),
}, section : 'NastranSolver')
