from setuptools import setup

setup(
    name="ampi",
    version="0.1.0",
    description="Adaptive Multi-Projection Index for Approximate Nearest Neighbor Search",
    package_dir={"": "src"},
    packages=["ampi"],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "numba>=0.50.0",
    ],
)
