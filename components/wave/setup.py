"""Build the independent, report-selected Wave FHT component."""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ["PYTORCH_ROCM_ARCH"] = os.environ.get("QUAROT_HIP_ARCHS", "gfx1201")

setup(
    name="quarot-wave-fht",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="quarot_wave_fht",
            sources=["binding.cpp", "wave.hip"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
