from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

#CUDA_HOME=/usr/local/cuda-12.4 python nca_cuda_compile.py install

setup(
    name='conv2d_cuda',
    ext_modules=[
        CUDAExtension('nca_cuda', [
            'nca_cuda.cu',
        ]),
        CUDAExtension('nca_cuda3d', [
            'nca_cuda3d.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
