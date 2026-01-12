from setuptools import setup, find_packages
import os

def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

setup(
    name="autoClip",
    version="0.1.0",
    description="Automated video highlighting pipeline",
    author="Code_Cradle",
    packages=find_packages(exclude=["autoClip", "autoClip.*", "tests*"]),
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "autoclip=shortclip.scripts.process_video:main",
        ],
    },
)
