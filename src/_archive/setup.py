from setuptools import setup, find_packages

setup(
    name="quantum_core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "torch>=2.0.0",
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.0",
        "matplotlib>=3.4.0",
        "paramiko",
        "asyncssh",
        "pytest",
        "pytest-asyncio"
    ],
    author="Greg",
    author_email="greg@windsurf.ai",
    description="WindSurf Quantum Core - Perfect Flow Engine",
    python_requires=">=3.8",
)
