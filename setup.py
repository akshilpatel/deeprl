import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="deeprl",
    version="0.0.1",
    author="Akshil Patel",
    author_email="akshilpatel11@gmail.com",
    description="Pytorch implementations of some Deep Reinforcement Learning algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akshilll/deeprl",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "gym", "torch", "pytest", "wandb", "matplotlib"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: English",
    ],
)
