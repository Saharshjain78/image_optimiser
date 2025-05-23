from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    required = f.read().splitlines()

# Read README for long description
with open('readme.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="neural_noise_segmentation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Neural Noise-Driven Image Segmentation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neural-noise-segmentation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=required,
    entry_points={
        "console_scripts": [
            "neural-noise-app=streamlit_app:main",
        ],
    },
    include_package_data=True,
)