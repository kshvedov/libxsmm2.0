import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

LIBXSMM_ROOT_PATH="/mnt/c/Users/cpenk/Documents/School/UO/2022_1_Winter/CIS_503_Thesis/libxsmm_2.0/"

setup(name='pcl_mlp',
      py_modules = ['pcl_mlp'],
      ext_modules=[CppExtension('pcl_mlp_ext', ['pcl_mlp_ext.cpp'], extra_compile_args=['-fopenmp', '-g', '-march=native'],
        include_dirs=['{}/include/'.format(LIBXSMM_ROOT_PATH)],
        library_dirs=['{}/lib/'.format(LIBXSMM_ROOT_PATH)],
        libraries=['xsmm'])],
      cmdclass={'build_ext': BuildExtension})

