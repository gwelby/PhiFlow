#!/usr/bin/env python3
"""
PhiFlow Setup Script
Sacred Geometry Programming Language
"""

from setuptools import setup, find_packages
import os
import sys

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "PhiFlow: Sacred Geometry Programming Language"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return ['matplotlib>=3.5.0', 'numpy>=1.21.0', 'Pillow>=8.3.0']

# Version information
VERSION = "1.0.0"

setup(
    name="phiflow",
    version=VERSION,
    author="Greg Welby",
    author_email="greg@phiflow.org",
    description="Sacred Geometry Programming Language for Consciousness-Enhanced Computing",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/gregwelby/PhiFlow",
    project_urls={
        "Documentation": "https://docs.phiflow.org",
        "Source": "https://github.com/gregwelby/PhiFlow",
        "Tracker": "https://github.com/gregwelby/PhiFlow/issues",
        "Funding": "https://github.com/sponsors/gregwelby",
        "Research Paper": "https://arxiv.org/abs/2025.phiflow",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "phiflow": [
            "examples/*.phi",
            "docs/*.md",
            "visualizations/.gitkeep",
        ],
    },
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "notebook>=6.4.0",
        ],
        "performance": [
            "numba>=0.56.0",
            "scipy>=1.7.0",
        ],
        "audio": [
            "sounddevice>=0.4.0",
            "librosa>=0.8.0",
    ],
    },
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "phiflow=phiflow.cli:main",
            "phi=phiflow.cli:main",
        ],
    },
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        
        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Other Audience",
        
        # Topic
        "Topic :: Artistic Software",
        "Topic :: Education",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Interpreters",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Other",
        
        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        
        # Environment
        "Environment :: Console",
        "Environment :: X11 Applications",
        "Environment :: MacOS X",
        "Environment :: Win32 (MS Windows)",
        
        # Natural Language
        "Natural Language :: English",
    ],
    keywords=[
        "sacred geometry",
        "golden ratio",
        "phi",
        "fibonacci",
        "consciousness computing",
        "domain specific language",
        "dsl",
        "visualization",
        "mathematics",
        "geometry",
        "art",
        "meditation",
        "healing",
        "frequency",
        "harmonic",
        "mandala",
        "flower of life",
        "merkaba",
        "platonic solids",
        "sri yantra",
        "torus",
        "chakra",
        "programming language",
        "interpreter",
        "sacred mathematics",
        "geometric patterns",
        "spiritual technology",
        "consciousness",
        "quantum",
        "holistic computing",
    ],
    license="MIT",
    zip_safe=False,
    platforms=["any"],
    
    # Additional metadata for PyPI
    maintainer="Greg Welby",
    maintainer_email="greg@phiflow.org",
    
    # Minimum Python version check
    cmdclass={},
    
    # Test suite
    test_suite="tests",
    tests_require=[
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
    ],
)

# Post-installation message
def print_installation_success():
    print("\n" + "="*60)
    print("🌟 PhiFlow Installation Complete! 🌟")
    print("="*60)
    print("Sacred Geometry Programming Language is ready!")
    print()
    print("Quick Start:")
    print("  phiflow examples/simple_meditation.phi")
    print()
    print("Documentation:")
    print("  https://docs.phiflow.org")
    print()
    print("Examples:")
    print("  phiflow --list-examples")
    print()
    print("Community:")
    print("  GitHub: https://github.com/gregwelby/PhiFlow")
    print("  Discord: https://discord.gg/phiflow")
    print()
    print("Thank you for joining the Sacred Geometry Revolution!")
    print("="*60)

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: PhiFlow requires Python 3.7 or higher")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    # Run setup
    setup()
    
    # Print success message if installation completed
    if len(sys.argv) > 1 and sys.argv[1] in ['install', 'develop']:
        print_installation_success()
