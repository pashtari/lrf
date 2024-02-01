#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="lsvd",
    version="0.0.1",
    description="A PyTorch implementation of LSVD",
    author="Pooya Ashtari and Pourya Behmandpoor",
    author_email="pooya.ash@gmail.com",
    url="https://github.com/pashtari/lsvd",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    keywords=[
        "machine learning",
        "deep learning",
        "image classification",
        "image compression",
        "singular value decomposition",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "torchvision",
        "hydra-core",
        "einops",
    ],
)
