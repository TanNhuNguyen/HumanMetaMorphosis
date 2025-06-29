#*********************************************************************************************************#
#********************************************* SUPPORTING LIBRARIES **************************************#
#*********************************************************************************************************#
from setuptools import setup, find_packages;

#*********************************************************************************************************#
#********************************************* MAIN FUNCTIONS ********************************************#
#*********************************************************************************************************#
setup(
    # The name of your package as it will appear on PyPI
    name="BodyShapeAnalyzer",
    # The current version of your package, following semantic versioning
    version="0.1.0",
    # Automatically find all packages and sub-packages in the source directory
    packages=find_packages(),
    # A short summary of what your package does
    description="A package for analyzing human body shape and BMI",
    # Your name as the package author
    author="Tan-Nhu",
    # A list of other Python packages your project depends on
    install_requires=[], # Leave empty for now, add dependencies as needed
    # Minimum Python version required to use this package
    python_requires=">=3.7"
)