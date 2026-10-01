"""Isaac Asimov installation."""

from __future__ import annotations

from pathlib import Path

import toml
from setuptools import find_packages, setup

EXTENSION_ROOT = Path(__file__).resolve().parent
EXTENSION_METADATA = toml.load(EXTENSION_ROOT / "config" / "extension.toml")["package"]

setup(
    name="isaac_asimov",
    version=EXTENSION_METADATA["version"],
    description=EXTENSION_METADATA["description"],
    author=EXTENSION_METADATA["author"],
    maintainer=EXTENSION_METADATA["maintainer"],
    url=EXTENSION_METADATA["repository"],
    license="BSD-3-Clause",
    packages=find_packages(),
    package_data={"isaac_asimov": ["tasks/locomotion/motions/*.npz"]},
    include_package_data=True,
    install_requires=["numpy<2", "rsl-rl-lib==5.0.1", "huggingface_hub"],
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    zip_safe=False,
)
