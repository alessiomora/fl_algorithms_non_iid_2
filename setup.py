import setuptools

with open("requirements.txt", "r", encoding="utf8") as fh:
    requirements = fh.read().split("\n")
    requirements = [r for r in requirements if r and r[0] != "-"]

# with open("version", "r") as f:
#     version = f.readline()
version = "1.0.0"

setuptools.setup(
    name="experiments_lib",
    version=version,
    author="Alessio Mora",
    author_email="alessio.mora@unibo.it",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)