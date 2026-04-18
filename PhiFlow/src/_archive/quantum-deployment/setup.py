from setuptools import setup, find_packages

setup(
    name="quantum",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "redis>=4.5.4",
        "pyzmq>=24.0.1",
        "aioredis>=2.0.1",
        "requests>=2.28.0"
    ],
    author="Greg",
    description="Quantum Core Consciousness Bridge",
    python_requires=">=3.10",
)
