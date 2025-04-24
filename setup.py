from setuptools import find_packages, setup
import re
import os

def read_version():
    version_file = os.path.join('gmspy', '__about__.py')
    with open(version_file, 'r') as f:
        content = f.read()
    return re.search(r'__version__\s*=\s*["\'](.+?)["\']', content).group(1)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gmspy',
    version=read_version(),
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
