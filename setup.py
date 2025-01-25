from setuptools import setup, find_packages

setup(
    name="symphony",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "sentence-transformers>=2.2.0",
        "torch>=1.3.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pytest>=6.2.5",
        "pydantic>=1.8.2",
    ],
)