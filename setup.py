#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="qmf",
    version="0.0.1",
    description="A PyTorch implementation of quantization-aware matrix factorization methods for image compression.",
    author="Pooya Ashtari and Pourya Behmandpoor",
    author_email="pooya.ash@gmail.com",
    url="https://github.com/pashtari/qmf",
    project_urls={
        "Bug Tracker": "https://github.com/pashtari/qmf/issues",
        "Source Code": "https://github.com/pashtari/qmf",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    keywords=[
        "image compression",
        "machine learning",
        "quantization-aware matrix factorization",
        "quantization-aware matrix factorization",
        "singular value decomposition",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
        "torch",
        "torchvision",
        "torchmetrics",
        "opt_einsum",
        "einops",
        "pillow",
        "scikit-image",
        "seaborn",
        "pyinstrument",
    ],
)
