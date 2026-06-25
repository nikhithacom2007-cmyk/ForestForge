from setuptools import setup, find_packages

setup(
    name="forestforge",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    author="Your Name",
    description="A Machine Learning package for Decision Trees and Random Forests built from scratch in NumPy.",
    python_requires=">=3.8",
)
