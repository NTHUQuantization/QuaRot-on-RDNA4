import os

from setuptools import find_packages, setup
import torch.utils.cpp_extension as torch_cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup_dir = os.path.dirname(os.path.realpath(__file__))

# Build for the project's primary target unless the caller requests others.
os.environ["PYTORCH_ROCM_ARCH"] = os.environ.get(
    "QUAROT_HIP_ARCHS", "gfx1201"
)

def remove_unwanted_pytorch_flags():
    flags = [
        "-D__HIP_NO_HALF_OPERATORS__=1",
        "-D__HIP_NO_HALF_CONVERSIONS__=1",
    ]

    for flag in flags:
        for flag_list in [
            torch_cpp_ext.COMMON_NVCC_FLAGS,
            torch_cpp_ext.COMMON_HIP_FLAGS,
        ]:
            try:
                while flag in flag_list:
                    flag_list.remove(flag)
            except AttributeError:
                pass


if __name__ == '__main__':
    remove_unwanted_pytorch_flags()
    setup(
        name='quarot',
        version='0.1.0',
        description='QuaRot W4A4KV4 inference for AMD RDNA 4',
        long_description=open(os.path.join(setup_dir, 'README.md'), encoding='utf-8').read(),
        long_description_content_type='text/markdown',
        url='https://github.com/NTHUQuantization/QuaRot-on-RDNA4',
        license='See LICENSE and THIRD_PARTY_NOTICES.md',
        license_files=['LICENSE', 'THIRD_PARTY_NOTICES.md', 'licenses/*.txt'],
        python_requires='>=3.10',
        packages=find_packages(include=['quarot', 'quarot.*', 'e2e', 'e2e.*']),
        include_package_data=True,
        exclude_package_data={'quarot': ['kernels/gemm_hip.hip', 'kernels/quant_hip.hip']},
        install_requires=[line.strip() for line in open(
            os.path.join(setup_dir, 'requirements.txt'), encoding='utf-8')
            if line.strip() and not line.lstrip().startswith('#')],
        ext_modules=[
            # PyTorch uses CUDAExtension for both CUDA and ROCm/HIP sources.
            CUDAExtension(
                name='quarot._HIP',
                sources=[
                    'quarot/kernels/bindings.cpp',
                    'quarot/kernels/gemm.hip',
                    'quarot/kernels/quant.hip',
                    'quarot/kernels/flashinfer.hip',
                    'quarot/kernels/fused_hip.hip',
                    'quarot/kernels/verification_preprocess.hip',
                    'quarot/kernels/verification_metadata.hip',
                ],
                include_dirs=[
                    os.path.join(setup_dir, 'quarot/kernels/include_hip'),
                ],
                extra_compile_args={
                    "cxx": [
                        "-O3",
                        "-std=c++17",
                    ],
                    "nvcc": [
                        "-O3",
                        # Host-side dispatch also needs this definition; HIP
                        # architecture macros are device-pass-only.
                        "-DQUAROT_BPRE_GFX12=1",
                        # PyTorch may add these flags after COMMON_HIP_FLAGS
                        # has been edited, so undefine them here as well.
                        "-U__HIP_NO_HALF_OPERATORS__",
                        "-U__HIP_NO_HALF_CONVERSIONS__",
                    ],
                }
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
