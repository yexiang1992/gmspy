from setuptools import find_packages, setup

from gmspy import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gmspy',
    version=__version__,
    description='Ground Motions Signal Processing for Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Yexiang Yan',
    author_email='yexiang_yan@outlook.com',
    url='https://github.com/yexiang1992',
    license='GPL Licence',
    keywords='Ground Motions Seismic IMs response spectra',
    platforms='any',
    python_requires='>=3.8',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=[
        'matplotlib', 'numpy', 'scipy', 'joblib', 'numba', 'rich'
    ],
    include_package_data=True,
)
