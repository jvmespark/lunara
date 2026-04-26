"""Lunara — Python package setup."""

from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
LONG_DESC = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="lunara",
    version="0.1.0",
    description="ML compiler: ONNX → StableHLO → MLIR linalg → Triton/PTX",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    author="Lunara Contributors",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=find_packages(exclude=("tests", "tests.*", "examples", "tools")),
    include_package_data=True,
    install_requires=[
        "torch>=2.1.0",
        "triton>=2.1.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "profiler": ["matplotlib>=3.5.0"],
        "onnx":     ["onnx>=1.14.0", "onnxruntime>=1.16.0"],
        "dev":      ["pytest>=7.0", "pytest-xdist", "pytest-cov",
                     "black", "ruff", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "lunara = lunara.driver:_main",
            "lunara-emit = lunara.codegen.triton_emitter:__main__",
            "lunara-tune = lunara.codegen.autotuner:__main__",
            "lunara-profile = lunara.profiler.nsight_profiler:__main__",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers",
    ],
)
