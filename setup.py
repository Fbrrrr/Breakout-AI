#!/usr/bin/env python3
"""
Setup script for Breakout AI using NEAT
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="breakout-ai-neat",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Breakout game AI using NEAT (NeuroEvolution of Augmenting Topologies)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Fbrrrr/breakout-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Games/Entertainment :: Arcade",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ai, neat, evolutionary-algorithm, game-ai, breakout, neural-networks, machine-learning",
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "analysis": [
            "jupyter>=1.0",
            "seaborn>=0.11",
            "pandas>=1.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "breakout-ai=BreakoutAI:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/Fbrrrr/breakout-ai/issues",
        "Source": "https://github.com/Fbrrrr/breakout-ai",
        "Documentation": "https://github.com/Fbrrrr/breakout-ai#readme",
    },
) 