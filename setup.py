"""Package configuration for elliptic-fraud-detection."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="elliptic-fraud-detection",
    version="0.1.0",
    description=(
        "Bitcoin transaction fraud detection using XGBoost, Random Forest, and SHAP "
        "interpretability on the Elliptic++ graph dataset."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "mlflow>=2.5.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "networkx>=3.1",
        "shap",
        "missingno>=0.5.2",
        "tqdm>=4.65.0",
        "joblib",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
        ],
    },
)
