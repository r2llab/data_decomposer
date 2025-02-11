from setuptools import setup, find_packages

setup(
    name="r2l",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "pyyaml",
        "sentence-transformers",
        "openai",
    ],
    python_requires=">=3.7",
)