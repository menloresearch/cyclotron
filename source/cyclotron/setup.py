"""Cyclotron installation."""

from __future__ import annotations

from pathlib import Path

import toml
from setuptools import find_packages, setup

EXTENSION_ROOT = Path(__file__).resolve().parent
EXTENSION_METADATA = toml.load(EXTENSION_ROOT / "config" / "extension.toml")["package"]

setup(
    name="cyclotron",
    version=EXTENSION_METADATA["version"],
    description=EXTENSION_METADATA["description"],
    author=EXTENSION_METADATA["author"],
    maintainer=EXTENSION_METADATA["maintainer"],
    url=EXTENSION_METADATA["repository"],
    license="BSD-3-Clause",
    packages=find_packages(),
    package_data={"cyclotron": ["tasks/locomotion/motions/*.npz", "tasks/locomotion/mdp/*.pyi"]},
    include_package_data=True,
    install_requires=["numpy>=2", "rsl-rl-lib==5.5.1", "huggingface_hub>=0.36,<2", "PyYAML>=6", "filelock>=3", "onnx", "onnxruntime", "onnxscript"],
    python_requires=">=3.12",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    zip_safe=False,
)
