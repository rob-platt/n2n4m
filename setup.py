from setuptools import setup, find_packages

VERSION = '0.0.2'
DESCRIPTION = 'Noise2Noise for Mars: Package for denoising and visualization of CRISM data L sensor data.'
LONG_DESCRIPTION = DESCRIPTION

# Setting up
setup(
    name="n2n4m",
    version=VERSION,
    author="Robert Platt",
    author_email="rp1818@ic.ac.uk",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    python_requires=">=3.10, <3.12",
    packages=find_packages(),
    package_data={
        "n2n4m": ["data/*"]
    },
    include_package_data=True,
    install_requires=[
        "torch >= 2.0",
        "scikit-learn >= 1.2",
        "ray >= 2",
        "pandas",
        "pytest",
        "ipykernel",
        "pyarrow",
        "ipywidgets",
        "crism_ml @ git+https://github.com/Banus/crism_ml.git@master#egg=crism_ml",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ]
)