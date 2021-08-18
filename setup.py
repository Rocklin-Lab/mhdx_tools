from setuptools import setup, find_packages
import os.path

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hdx_limit",
    version="0.2.0",
    description="Tools for analysis of LC-IMS-MS data represented as 3D tensors.",

    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/Rocklin-Lab/hdx_limit",

    author="Robert Wes Ludwig, Sugyan Dixit, Állan Ferrari, Gabriel Rocklin",
    author_email="robert.wes.ludwig@gmail.com, suggie@northwestern.edu, ajrferrari@northwestern.edu, grocklin@gmail.com",

    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",

        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",

        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by "pip install". See instead "python_requires" below.
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],

    keywords="scientific-computing, mass spectrometry, liquid chromatography, ion mobility separation, hydrogen exchange, deuterium, HDX",

    #packages=find_packages(),

    python_requires=">=3.6, <4",

    install_requires=["pandas>=1.2<2",
                      "scikit-learn>=0.24<0.3",
                      "matplotlib>=3.3<4",
                      "scipy>=1.6<2",
                      "peakutils>=1.3<2",
                      "pyyaml>=5.4<6",
                      "seaborn>=0.11<0.12",
                      "nn-fac>=0.1.2<0.2",
                      "molmass>=2020.1.1",
                      "ipdb>=0.13<0.14",
                      ],  # external packages as dependencies

    project_urls={
        "Source": "https://github.com/Rocklin-Lab/hdx_limit",
        "Gabriel Rocklin Lab": "www.rocklinlab.org",
    },

    packages = ["core", "preprocessing", "pipeline"]

)