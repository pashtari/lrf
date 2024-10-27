#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="lrf",
    version="0.0.1",
    description="A PyTorch implementation of low-rank factorization methods for image compression.",
    author="Pooya Ashtari and Pourya Behmandpoor",
    author_email="pooya.ash@gmail.com",
    url="https://github.com/pashtari/lrf",
    project_urls={
        "Bug Tracker": "https://github.com/pashtari/lrf/issues",
        "Source Code": "https://github.com/pashtari/lrf",
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
        "low-rank factorization",
        "integer matrix factorization",
        "singular value decomposition",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "torchvision",
        "opt_einsum",
        "einops",
        "pillow",
        "skimage",
        "seaborn",
        "pyinstrument",
    ],
)
