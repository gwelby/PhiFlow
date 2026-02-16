from setuptools import setup, find_packages

setup(
    name="quantum",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "redis>=4.5.4",
        "pyzmq>=24.0.1",
        "aioredis>=2.0.1",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "torch>=2.0.0"
    ],
    author="Greg",
    description="Quantum Core Consciousness Bridge",
    python_requires=">=3.11",
)
