#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='crcf',
    python_requires='>=3.7',
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
    install_requires=["numpy", "graphviz"],
    version='0.0.2',
    author='J. Marcus Hughes',
    author_email='hughes.jmb@gmail.com',
    packages=find_packages(),
    url='',
    license='LICENSE.txt',
    description='Combination Robust Cut Forests',
    long_description=open('Readme.md').read(),
    long_description_content_type="text/markdown",
)
