# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# python .\setup.py build_ext --inplace
setup(
    name='culina',
    ext_modules=[
        CUDAExtension(
            name='custom_op',
            sources=['./cuda/custom_op.cu'],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        ),
        CUDAExtension(
            name='cuda_reduce',
            sources=['./cuda/warp_reduce.cu'],
            include_dirs=['../../'],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        ),
        CUDAExtension(
            name='cuda_scan',
            sources=['./cuda/scan.cu'],
            include_dirs=['../../'],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)