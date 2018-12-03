#!/usr/bin/env python

from distutils.core import setup

setup(
    name='oiforests',
    version='0.0.1',
    author='J. Marcus Hughes',
    author_email='hughes.jmb@gmail.com',
    packages=['oiforests'],
    url='',
    license='LICENSE.txt',
    description='Online Isolation Forests',
    long_description=open('Readme.md').read(),
    install_requires=["numpy"],
    test_suite="tests"
)