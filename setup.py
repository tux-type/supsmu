from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np
import sys

def get_compile_args():
    if sys.platform.startswith("win"):
        # MSVC
        return ["/O2", "/arch:AVX2"]
    else:
        # GCC/Clang
        return ["-O3"]

extensions = [
    Extension(
        "supsmu._supsmu",
        ["src/supsmu/_supsmu.pyx", "src/supsmu/supsmu.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=get_compile_args(),
    )
]

setup(
    ext_modules=cythonize(extensions, language_level="3")
)
