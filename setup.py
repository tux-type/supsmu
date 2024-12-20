from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "supsmu._supsmu",
        ["src/supsmu/_supsmu.pyx", "src/supsmu/supsmu.c"],
        include_dirs=[np.get_include()]
    )
]

setup(
    ext_modules=cythonize(ext_modules)
)
