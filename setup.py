# TODO
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="teg",  # Replace with your own PACKAGE NAME. For the love of god..
    version="0.0.1",
    author="Sai Bangaru",
    author_email="sbangaru@mit.edu",
    description="Teg: A framework for generalized differentiable computation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
