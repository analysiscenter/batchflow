"""
BatchFlow helps you conveniently work with random or sequential batches of your data
and define data processing and machine learning workflows even for datasets that do not fit into memory.

Documentation - https://analysiscenter.github.io/batchflow/
"""
import re
from setuptools import setup, find_packages


with open('batchflow/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)


with open('README.md', 'r') as f:
    long_description = f.read()


setup(
    name='batchflow',
    packages=find_packages(exclude=['examples']),
    version=version,
    url='https://github.com/analysiscenter/batchflow',
    license='Apache License 2.0',
    author='Roman Kh at al',
    author_email='rhudor@gmail.com',
    description='A framework for fast data processing and ML models training',
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    platforms='any',
    install_requires=[
        'urllib3>=1.25',
        'numpy>=1.10',
        'pandas>=0.24',
        'dill>=0.2.7',
        'tqdm>=4.19.7',
        'scipy>=0.19.1',
        'scikit-image>=0.13.1',
        'matplotlib>=3.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering'
    ],
)
