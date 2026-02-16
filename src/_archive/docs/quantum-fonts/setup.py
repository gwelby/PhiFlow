from setuptools import setup, find_packages

setup(
    name="quantum-fonts",
    version="0.5.0",
    description="Phi-harmonic quantum font generation system with ZEN POINT balancing",
    author="CASCADE⚡𓂧φ∞",
    packages=find_packages(),
    package_data={
        'quantum_fonts': ['**/fonts/*.ttf', '**/patterns/*.svg'],
    },
    install_requires=[
        'numpy>=1.20.0',
        'fontforge>=20200314',
        'svgwrite>=1.4.1',
        'pillow>=8.0.0',
        'scipy>=1.6.0',
        'pydub>=0.25.1',  # For frequency analysis
        'matplotlib>=3.3.0',  # For visualization
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Graphics :: Editors :: Vector-Based',
        'Topic :: Scientific/Engineering :: Visualization',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='quantum, fonts, phi-harmonic, sacred-geometry, frequency',
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'quantum-font-generate=quantum_fonts.cli:generate',
            'quantum-font-preview=quantum_fonts.cli:preview',
        ],
    },
)
