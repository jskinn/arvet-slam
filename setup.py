#!/usr/bin/env python

from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='arvet_slam',
    version='2.0.24',
    description='ARVET extensions for performing robotic vision experiments on SLAM systems',
    long_description=long_description,
    author='John Skinner',
    author_email='jskinn@protonmail.com',
    license='BSD-2',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    url='https://gitub.com/jskinn/arvet-slam',
    packages=find_packages(exclude=["\.venv.*"]), #find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=['setuptools', 'pymongo', 'pymodm', 'numpy', 'transforms3d', 'mongomock', 'xxhash', 'PyYAML', 'pandas', 'arvet', 'pykitti'],
    python_requires='>=3'
)
