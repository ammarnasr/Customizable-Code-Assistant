import pathlib
import setuptools

setuptools.setup(
    name="enigma_ai",
    version="0.1.1",
    description="Custom code assistant, get the data and train your code assistant",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://CMC-Project.github.io/custom-code-assistant",
    author="Enigma AI",
    license="MIT",
    project_urls={
        "Source": "https://cmcp.github.io/custom-code-assistant",
        "Documentation": "https://cmcp.github.io/custom-code-assistant",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Utilities",
    ],
    python_requires=">=3.9,<3.13",
    install_requires=[
        "pandas",
        "requests",
        "tqdm",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "enigma_ai=enigma_ai.cli:main",
        ],
    },
)
