import os
import shlex

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def split_arches(value):
    return [x.strip() for x in value.replace(",", ";").split(";") if x.strip()]


hip_arches = split_arches(os.environ.get("HADACORE_HIP_ARCHS", "gfx1201"))
os.environ.setdefault("PYTORCH_ROCM_ARCH", ";".join(hip_arches))

hip_flags = [
    "-O3",
    "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
]
if os.environ.get("HADACORE_ENABLE_EXPERIMENTAL_WMMA", "1") == "1":
    hip_flags.append("-DHADACORE_ENABLE_EXPERIMENTAL_WMMA=1")
if os.environ.get("HADACORE_FORCE_GFX12_WMMA", "1") == "1":
    hip_flags.append("-DHADACORE_FORCE_GFX12_WMMA=1")
hip_flags.extend(shlex.split(os.environ.get("HADACORE_HIP_EXTRA_FLAGS", "")))

for arch in hip_arches:
    hip_flags.append(f"--offload-arch={arch}")

setup(
    name="fast_hadamard_transform",
    ext_modules=[
        CUDAExtension(
            name="fast_hadamard_transform",
            sources=[
                "binding.hip",
                "hadacore_for_hip.hip",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": hip_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
