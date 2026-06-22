from setuptools import find_packages, setup

setup(
    name="equity-market-dashboard",
    version="1.0.0",
    description="Equity Market Analysis Dashboard — PySpark + FastAPI + Interactive UI",
    author="Satija",
    python_requires=">=3.10",
    packages=find_packages(include=["src*", "api*"]),
    install_requires=[
        "fastapi>=0.111",
        "uvicorn[standard]>=0.29",
        "pydantic>=2.7",
        "numpy>=1.26",
        "pandas>=2.2",
        "scipy>=1.13",
        "loguru>=0.7",
        "pyyaml>=6.0",
    ],
    extras_require={
        "spark": ["pyspark>=3.5"],
        "dev": ["pytest", "pytest-cov", "black", "isort", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "equity-api=api.main:app",
            "equity-datagen=data.sample_generator:generate",
        ]
    },
)
