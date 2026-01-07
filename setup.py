"""
Setup configuration for Liver Disease Analysis package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="liver-disease-analysis",
    version="2.0.0",
    author="Felipe Breseghello",
    author_email="",
    description="Machine Learning project for liver disease prediction using clinical data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fbreseghello/Liver-Disease-Analysis-CNN",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ]
    },
    entry_points={
        "console_scripts": [
            "train-liver-model=train_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.csv"],
    },
    keywords=[
        "machine-learning",
        "healthcare",
        "liver-disease",
        "hepatitis",
        "medical-ai",
        "classification",
        "ensemble-learning",
        "optuna"
    ],
    project_urls={
        "Bug Reports": "https://github.com/fbreseghello/Liver-Disease-Analysis-CNN/issues",
        "Source": "https://github.com/fbreseghello/Liver-Disease-Analysis-CNN",
    },
)
