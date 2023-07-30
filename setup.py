from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="v.0.0.1",
    author="dibyendubiswas1998",
    author_email="dibyendubiswas1998@gmail.com",
    description="News Short Web App",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dibyendubiswas1998/NewsShort.git",
    packages=["src"],
    license="GNU",
    python_requires=">=3.10",
    install_requires=[
        'pandas',
        'scikit-learn',
        'nltk',
        'tensorflow',
        'transformers'
    ]
)

