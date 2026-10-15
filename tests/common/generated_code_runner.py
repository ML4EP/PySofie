"""Test helper: compile and run a header emitted by SOFIE for output validation.

The package itself only emits the C++ header. The tests write a small CMake project around it (a main
that reads the inputs, runs Session::infer and writes the output), build it with CMake, which finds the
compiler, BLAS and the CUDA toolkit, and run the executable in a separate process. The check of the
results is in the tests.
"""

import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

import sofie

_tests_dir = os.path.dirname(os.path.abspath(__file__))


def generated_dir(*parts):
    return os.path.join(_tests_dir, "..", "..", "generated", *parts)


def fresh_generated_dir(*parts):
    path = generated_dir(*parts)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path

# The generated CPU code is self-contained. The generated alpaka code still includes
# SOFIE/SOFIE_common.hxx and needs libSOFIE_core, so the sources of SOFIE (the submodule) are needed.
_sofie_include = os.path.join(_tests_dir, "..", "..", "SOFIE", "core", "inc")

_sanitize = os.environ.get("SOFIE_TEST_SANITIZE", "1") != "0"
_jobs = str(os.cpu_count() or 1)
_compile_timeout = float(os.environ.get("SOFIE_TEST_COMPILE_TIMEOUT", "900"))  # seconds
_run_timeout = float(os.environ.get("SOFIE_TEST_RUN_TIMEOUT", "120"))  # seconds

# tensor types of the inputs: C type, numpy type
_input_types = {
    "FLOAT": ("float", np.float32),
    "INT64": ("int64_t", np.int64),
    "INT32": ("int32_t", np.int32),
    "BOOL": ("bool", np.bool_),
}


# ---------------------------------------------------------------------------------------
# the executable
# ---------------------------------------------------------------------------------------
# Reading and writing of the inputs and the output: raw binary files.
_io = r"""
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

template <class T>
std::vector<T> read_exact(const char *path, std::size_t count)
{{
   std::FILE *f = std::fopen(path, "rb");
   if (!f)
      throw std::runtime_error(std::string("cannot open ") + path);
   std::vector<T> data(count);
   const std::size_t n = std::fread(data.data(), sizeof(T), count, f);
   char extra;
   const bool more = std::fread(&extra, 1, 1, f) == 1;
   std::fclose(f);
   if (n != count || more)
      throw std::runtime_error(std::string("the input file ") + path + " does not have exactly " +
                               std::to_string(count) + " elements");
   return data;
}}

static void write_all(const char *path, const float *data, std::size_t count)
{{
   std::FILE *f = std::fopen(path, "wb");
   if (!f)
      throw std::runtime_error(std::string("cannot open ") + path);
   const std::size_t n = std::fwrite(data, sizeof(float), count, f);
   std::fclose(f);
   if (n != count)
      throw std::runtime_error(std::string("cannot write ") + path);
}}
"""

# arguments of the executable: the weight file, the output file, then one file per input
_cpu_main = (
    r"""
#include "{header}"
"""
    + _io
    + r"""
int main(int argc, char **argv)
{{
   if (argc != {n_args}) {{
      std::fprintf(stderr, "error: expected {n_args} arguments (weights, output, inputs), got %d\n", argc);
      return 2;
   }}
   try {{
      {ns}::Session session(argv[1]);
{read_inputs}
      const auto result = session.infer({call_args});
      if (result.size() != {output_size})
         throw std::runtime_error("the model returned " + std::to_string(result.size()) + " values, expected {output_size}");
      write_all(argv[2], result.data(), result.size());
   }} catch (const std::exception &e) {{
      std::fprintf(stderr, "error: %s\n", e.what());
      return 2;
   }}
   return 0;
}}
"""
)

# the same for the GPU: the inputs are copied to the device and the output back
_cuda_main = (
    r"""
#include "{header}"
#include <array>
"""
    + _io
    + r"""
using Session = {ns}::Session<alpaka::TagGpuCudaRt>;
using Buf = Session::BufF1D;

int main(int argc, char **argv)
{{
   if (argc != {n_args}) {{
      std::fprintf(stderr, "error: expected {n_args} arguments (weights, output, inputs), got %d\n", argc);
      return 2;
   }}
   try {{
      Session session(argv[1]);
      const std::size_t sizes[] = {{{sizes}}};
{read_inputs}
      // copy the inputs to the device
      std::vector<Buf> in_bufs;
      const float *host_inputs[] = {{{host_pointers}}};
      for (std::size_t i = 0; i < {n_inputs}; ++i) {{
         in_bufs.push_back(alpaka::allocBuf<float, Session::Idx>(session.devAcc, Session::Ext1D::all(Session::Idx{{sizes[i]}})));
         auto host = alpaka::createView(session.hostAcc, host_inputs[i], Session::Ext1D::all(Session::Idx{{sizes[i]}}));
         alpaka::memcpy(session.queue, in_bufs.back(), host);
      }}
      alpaka::wait(session.queue);
      std::array<Session::ViewConstF1D, {n_inputs}> in{{{in_views}}};

      const std::size_t out_size = {output_size};
      Buf out_buf = alpaka::allocBuf<float, Session::Idx>(session.devAcc, Session::Ext1D::all(Session::Idx{{out_size}}));
      std::array<Session::ViewF1D, 1> out{{Session::ViewF1D{{alpaka::getPtrNative(out_buf), session.devAcc, Session::Ext1D::all(Session::Idx{{out_size}})}}}};
      session.infer(std::span<Session::ViewConstF1D const>(in), std::span<Session::ViewF1D>(out));
      alpaka::wait(session.queue);

      std::vector<float> result(out_size);
      auto host_out = alpaka::createView(session.hostAcc, result.data(), Session::Ext1D::all(Session::Idx{{out_size}}));
      alpaka::memcpy(session.queue, host_out, out_buf);
      alpaka::wait(session.queue);
      write_all(argv[2], result.data(), out_size);
   }} catch (const std::exception &e) {{
      std::fprintf(stderr, "error: %s\n", e.what());
      return 2;
   }}
   return 0;
}}
"""
)


class compiled_model:
    """A compiled model, run in a separate process for each inference: float output, one output tensor."""

    def __init__(self, executable, build_dir, weight_file, input_dtypes, input_sizes, output_shape):
        self._executable = executable
        self._build_dir = build_dir
        self._weight_file = weight_file
        self._input_dtypes = input_dtypes
        self._input_sizes = input_sizes
        self.output_shape = list(output_shape)

    def infer(self, *inputs):
        """Run inference. Returns the flattened output as a float32 numpy array."""
        if len(inputs) != len(self._input_dtypes):
            raise ValueError(f"the model has {len(self._input_dtypes)} inputs, got {len(inputs)}")
        arrays = []
        for i, (x, dtype, size) in enumerate(zip(inputs, self._input_dtypes, self._input_sizes)):
            array = np.ascontiguousarray(x, dtype=dtype)
            if array.size != size:
                raise ValueError(f"input {i} has {array.size} elements, the model expects {size}")
            arrays.append(array)

        with tempfile.TemporaryDirectory(prefix="sofie_run_") as tmp:
            input_files = []
            for i, array in enumerate(arrays):
                path = os.path.join(tmp, f"input{i}.bin")
                array.tofile(path)
                input_files.append(path)
            output_file = os.path.join(tmp, "output.bin")
            try:
                result = subprocess.run(
                    [self._executable, self._weight_file, output_file, *input_files],
                    capture_output=True,
                    text=True,
                    timeout=_run_timeout,
                )
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(f"the generated code did not finish in {_run_timeout:.0f} s") from e
            if result.returncode != 0:
                raise RuntimeError(
                    f"the generated code failed (exit status {result.returncode}):\n{result.stderr[-6000:]}"
                )
            output = np.fromfile(output_file, dtype=np.float32)
        expected = int(np.prod(self.output_shape))
        if output.size != expected:
            raise RuntimeError(f"the model returned {output.size} values, expected {expected}")
        return output

    def close(self):
        shutil.rmtree(self._build_dir, ignore_errors=True)

    def __del__(self):
        self.close()


def _write(directory, name, text):
    with open(os.path.join(directory, name), "w") as f:
        f.write(text)


def _checked_path(path):
    """The absolute path of a file, if it can be written into C++ source safely."""
    path = os.path.abspath(path)
    if not re.fullmatch(r"[A-Za-z0-9_./+\-]+", path):
        raise ValueError(f"the path {path!r} has characters that are not allowed in the include of generated code")
    return path


def _namespace(header_path):
    with open(header_path) as f:
        match = re.search(r"^namespace (SOFIE_\w+)\s*\{", f.read(), re.M)
    if match is None:
        raise RuntimeError(f"cannot find the namespace of the generated code in {header_path}")
    return match.group(1)


def _run(command, what, timeout=None):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout or _compile_timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"{what}: did not finish in {timeout or _compile_timeout:.0f} s") from e
    if result.returncode != 0:
        raise RuntimeError(f"{what}:\n{(result.stdout + result.stderr)[-6000:]}")


def _cmake():
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("cmake is needed to compile the generated code")
    return cmake


def _cmake_build(project_dir, what, definitions=(), target=None):
    """Configure and build the CMake project in `project_dir`; returns the build directory."""
    build_dir = os.path.join(project_dir, "build")
    _run([_cmake(), "-S", project_dir, "-B", build_dir, *[f"-D{d}" for d in definitions]], f"{what} (cmake configure)")
    command = [_cmake(), "--build", build_dir, "--parallel", _jobs]
    if target:
        command += ["--target", target]
    _run(command, f"{what} (cmake build)")
    return build_dir


_cpu_cmake = """
cmake_minimum_required(VERSION 3.18)
project(sofie_model LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(BLAS REQUIRED)
add_executable(model main.cpp)
target_link_libraries(model PRIVATE BLAS::BLAS)
if(SOFIE_TEST_SANITIZE)
  target_compile_options(model PRIVATE -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
  target_link_options(model PRIVATE -fsanitize=address,undefined)
else()
  target_compile_options(model PRIVATE -O2 -g)
endif()
"""


def compile_model(rmodel, header_path, batch_size=-1):
    """Generate the inference code of `rmodel` into `header_path` and compile it for the CPU."""
    header_path = _checked_path(header_path)
    rmodel.generate(batch_size=batch_size)
    rmodel.output_generated(header_path)
    namespace = _namespace(header_path)

    names = rmodel.get_input_tensor_names()
    types = [_input_types[rmodel.get_tensor_type(n).name] for n in names]
    sizes = [int(np.prod(rmodel.get_tensor_shape(n))) for n in names]
    output_size = int(np.prod(rmodel.get_tensor_shape(rmodel.get_output_tensor_names()[0])))
    read_inputs = "\n".join(
        f"      const auto in{i} = read_exact<{c}>(argv[{3 + i}], {size});" for i, ((c, _), size) in enumerate(zip(types, sizes))
    )
    source_text = _cpu_main.format(
        header=header_path,
        ns=namespace,
        n_args=3 + len(names),
        read_inputs=read_inputs,
        call_args=", ".join(f"in{i}.data()" for i in range(len(names))),
        output_size=output_size,
    )

    build_dir = tempfile.mkdtemp(prefix="sofie_build_")
    try:
        _write(build_dir, "main.cpp", source_text)
        _write(build_dir, "CMakeLists.txt", _cpu_cmake)
        built = _cmake_build(
            build_dir,
            f"Error compiling {header_path}",
            [f"SOFIE_TEST_SANITIZE={'ON' if _sanitize else 'OFF'}"],
            target="model",
        )
        executable = os.path.join(built, "model")
    except BaseException:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise
    weight_file = header_path[: -len(".hxx")] + ".dat"
    return compiled_model(executable, build_dir, weight_file, [d for _, d in types], sizes, rmodel.get_tensor_shape(rmodel.get_output_tensor_names()[0]))


# ---------------------------------------------------------------------------------------
# heterogeneous code (alpaka)
# ---------------------------------------------------------------------------------------
def alpaka_dependencies():
    """Directories of the alpaka and sofieBLAS sources (fetched by ./install.sh dev --alpaka --arch cuda), or None."""
    deps = os.environ.get("SOFIE_DEPS_DIR", os.path.join(_tests_dir, "..", "..", ".deps"))
    alpaka = os.environ.get("SOFIE_ALPAKA_DIR", os.path.join(deps, "alpaka"))
    blas = os.environ.get("SOFIE_BLAS_DIR", os.path.join(deps, "sofieBLAS"))
    if os.path.isfile(os.path.join(alpaka, "include", "alpaka", "alpaka.hpp")) and os.path.isfile(
        os.path.join(blas, "include", "sofieBLAS", "sofieBLAS.hpp")
    ):
        return os.path.abspath(alpaka), os.path.abspath(blas)
    return None


def alpaka_enabled():
    """Whether the alpaka tests can run: a build for an alpaka backend (./install.sh dev --alpaka --arch cuda) and the sources of alpaka and sofieBLAS."""
    return bool(sofie.with_alpaka) and alpaka_dependencies() is not None


_alpaka_check_cmake = """
cmake_minimum_required(VERSION 3.18)
project(sofie_alpaka_check LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(check OBJECT check.cpp)
target_include_directories(check PRIVATE "${SOFIE_INCLUDE}" "${ALPAKA_INCLUDE}" "${SOFIEBLAS_INCLUDE}")
target_compile_definitions(check PRIVATE ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED ALPAKA_HAS_STD_ATOMIC_REF)
"""


def check_alpaka_header(header_path):
    """Compile the generated alpaka header (without running it) against alpaka and sofieBLAS.

    Only code that does not depend on the accelerator tag is checked: the Session class is a
    template, and the CPU backend of sofieBLAS cannot be used with the generated code (its
    constructor takes a blocking queue and it has no addOperationConfig), so the code cannot be
    instantiated without a GPU backend (CUDA or HIP) and its toolchain.
    """
    alpaka_dir, blas_dir = alpaka_dependencies()
    header_path = _checked_path(header_path)
    project = tempfile.mkdtemp(prefix="sofie_check_")
    try:
        _write(project, "check.cpp", f'#include "{header_path}"\n')
        _write(project, "CMakeLists.txt", _alpaka_check_cmake)
        _cmake_build(
            project,
            f"Error compiling {header_path}",
            [
                f"SOFIE_INCLUDE={_sofie_include}",
                f"ALPAKA_INCLUDE={os.path.join(alpaka_dir, 'include')}",
                f"SOFIEBLAS_INCLUDE={os.path.join(blas_dir, 'include')}",
            ],
        )
    finally:
        shutil.rmtree(project, ignore_errors=True)


def _nvcc():
    for candidate in (os.environ.get("CUDACXX"), os.environ.get("NVCC"), "/usr/local/cuda/bin/nvcc", shutil.which("nvcc")):
        if candidate and os.path.exists(candidate):
            return os.path.realpath(candidate)
    return None


def _cuda_definitions():
    """The CMake definitions that point it at the CUDA toolkit of the nvcc that was found."""
    nvcc = _nvcc()
    cuda_home = os.path.dirname(os.path.dirname(nvcc))
    return [f"CMAKE_CUDA_COMPILER={nvcc}", f"CUDAToolkit_ROOT={cuda_home}"]


_cuda_probe_cmake = """
cmake_minimum_required(VERSION 3.24)
project(sofie_cuda_probe LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES native)
find_package(CUDAToolkit REQUIRED)
add_executable(probe probe.cu)
target_link_libraries(probe PRIVATE CUDA::cudart)
"""

_cuda_probe = r"""
#include <cuda_runtime.h>
int main() { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess && n > 0 ? 0 : 1; }
"""


def cuda_available():
    """Whether the alpaka tests can run for CUDA: a build for the cuda backend, alpaka and sofieBLAS, and a usable CUDA toolkit and GPU."""
    if not hasattr(cuda_available, "result"):
        result = False
        if getattr(sofie, "alpaka_backend", None) == "cuda" and _nvcc() and alpaka_enabled() and shutil.which("cmake"):
            project = tempfile.mkdtemp(prefix="sofie_cuda_probe_")
            try:
                _write(project, "probe.cu", _cuda_probe)
                _write(project, "CMakeLists.txt", _cuda_probe_cmake)
                build = _cmake_build(project, "cuda probe", _cuda_definitions(), target="probe")
                result = subprocess.run([os.path.join(build, "probe")], capture_output=True).returncode == 0
            except RuntimeError:
                result = False
            finally:
                shutil.rmtree(project, ignore_errors=True)
        cuda_available.result = result
    return cuda_available.result


_cuda_cmake = """
cmake_minimum_required(VERSION 3.24)
project(sofie_model LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
find_package(CUDAToolkit REQUIRED)
add_executable(model main.cu)
target_include_directories(model PRIVATE "${SOFIE_INCLUDE}" "${ALPAKA_INCLUDE}" "${SOFIEBLAS_INCLUDE}")
target_compile_definitions(model PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED ALPAKA_HAS_STD_ATOMIC_REF)
target_compile_options(model PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr --use_fast_math -Wno-deprecated-gpu-targets -O2>)
# libSOFIE_core, which the generated code still needs (ReadTensorFromStream)
target_link_directories(model PRIVATE "${SOFIE_LIB_DIR}")
target_link_libraries(model PRIVATE SOFIE_core CUDA::cudart CUDA::cublas CUDA::cublasLt)
set_target_properties(model PROPERTIES BUILD_RPATH "${SOFIE_LIB_DIR}")
"""


def compile_model_cuda(rmodel, header_path, batch_size=-1):
    """Generate the alpaka code of `rmodel`, compile it for the GPU with CMake and CUDA.

    Float inputs and a single output only. The sanitizers do not apply to device code: the code is
    still run in a separate process.
    """
    alpaka_dir, blas_dir = alpaka_dependencies()
    header_path = _checked_path(header_path)
    rmodel.generate_gpu_alpaka(batch_size=batch_size)
    rmodel.output_generated(header_path)
    namespace = _namespace(header_path)

    names = rmodel.get_input_tensor_names()
    if not all(rmodel.get_tensor_type(n).name == "FLOAT" for n in names):
        raise ValueError("only float inputs are supported")
    sizes = [int(np.prod(rmodel.get_tensor_shape(n))) for n in names]
    output_shape = rmodel.get_tensor_shape(rmodel.get_output_tensor_names()[0])
    source_text = _cuda_main.format(
        header=header_path,
        ns=namespace,
        n_args=3 + len(names),
        n_inputs=len(names),
        sizes=", ".join(str(n) for n in sizes),
        read_inputs="\n".join(f"      const auto in{i} = read_exact<float>(argv[{3 + i}], {n});" for i, n in enumerate(sizes)),
        host_pointers=", ".join(f"in{i}.data()" for i in range(len(names))),
        in_views=", ".join(
            "Session::ViewConstF1D{alpaka::getPtrNative(in_bufs[%d]), session.devAcc, Session::Ext1D::all(Session::Idx{%d})}" % (i, n)
            for i, n in enumerate(sizes)
        ),
        output_size=int(np.prod(output_shape)),
    )

    build_dir = tempfile.mkdtemp(prefix="sofie_cuda_build_")
    try:
        _write(build_dir, "main.cu", source_text)
        _write(build_dir, "CMakeLists.txt", _cuda_cmake)
        built = _cmake_build(
            build_dir,
            f"Error compiling {header_path}",
            [
                *_cuda_definitions(),
                f"SOFIE_INCLUDE={_sofie_include}",
                f"ALPAKA_INCLUDE={os.path.join(alpaka_dir, 'include')}",
                f"SOFIEBLAS_INCLUDE={os.path.join(blas_dir, 'include')}",
                f"SOFIE_LIB_DIR={os.path.dirname(sofie._sofie.__file__)}",
            ],
            target="model",
        )
        executable = os.path.join(built, "model")
    except BaseException:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise
    weight_file = header_path[: -len(".hxx")] + ".dat"
    return compiled_model(executable, build_dir, weight_file, [np.float32] * len(names), sizes, output_shape)
