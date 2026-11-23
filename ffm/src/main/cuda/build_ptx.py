"""Compile ffm/src/main/cuda/*.cu to PTX resources with NVRTC (development tool).

The PTX is committed under ffm/src/main/resources/scalatorch/, so neither nvcc nor NVRTC is needed at run time.

usage: python build_ptx.py [half]
"""
import ctypes, glob, os, sys

CUDA = next(p for p in sorted(glob.glob("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.*"), reverse=True))
os.add_dll_directory(CUDA + "/bin")
nvrtc = ctypes.CDLL(next(glob.iglob(CUDA + "/bin/nvrtc64_*_0.dll")))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "resources", "scalatorch")


def check(code, what):
    if code != 0:
        raise SystemExit(f"{what} failed with NVRTC status {code}")


def compile_file(name):
    source = open(os.path.join(HERE, name + ".cu"), "rb").read()
    program = ctypes.c_void_p()
    check(nvrtc.nvrtcCreateProgram(ctypes.byref(program), source, (name + ".cu").encode(), 0, None, None), "create")
    options = [b"--gpu-architecture=compute_75", b"--std=c++17", b"-default-device",
               b"--include-path=" + (CUDA + "/include").encode()]
    array = (ctypes.c_char_p * len(options))(*options)
    status = nvrtc.nvrtcCompileProgram(program, len(options), array)
    size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(program, log)
    if log.value.strip():
        print(log.value.decode(errors="replace"))
    check(status, "compile " + name)
    check(nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size)), "ptx size")
    ptx = ctypes.create_string_buffer(size.value)
    check(nvrtc.nvrtcGetPTX(program, ptx), "get ptx")
    path = os.path.join(OUT, name + ".ptx")
    with open(path, "wb") as f:
        f.write(ptx.value.replace(b"\r\n", b"\n") + b"\n")
    print(f"wrote {path} ({len(ptx.value)} bytes)")


for target in (sys.argv[1:] or ["half"]):
    compile_file(target)
