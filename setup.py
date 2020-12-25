# TODO
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="teg",
    version="0.0.1",
    author="Jesse Michel, Sai Bangaru",
    author_email="sbangaru@mit.edu",  # Apparently can't put two email IDs
    description="Teg: A framework for generalized differentiable computation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    package_data={'': ['data/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'pybind11',
        'numpy>=1.14.5'
    ]
)
