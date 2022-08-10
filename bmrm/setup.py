from setuptools import Extension, setup
from Cython.Build import cythonize

source_files = ['bridge.pyx', 'libqp_splx.c']

extensions = [Extension("libqp_bridge", source_files)]

setup(
    ext_modules=cythonize(extensions)
)
