from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("pyport.pyx", language_level = "3")
)