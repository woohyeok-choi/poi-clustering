import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='poi-clustering',
    version='0.0.2',
    author='Woohyeok Choi',
    author_email='woohyeok.choi@kaist.ac.kr',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/woohyeok-choi/poi-clustering',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn'
    ],
    zip_safe=False,
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering"
    ]
)