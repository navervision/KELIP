"""
KELIP
Copyright (c) 2022-present NAVER Corp.
Apache-2.0
"""
import os
import pkg_resources
from setuptools import setup, find_packages

setup(
    name='kelip',
    version='0.1.0',
    description='easy to use KECLIP library',
    author='NAVER Corp.',
    author_email='dl_visionresearch@navercorp.com',
    url='https://github.com/navervision/KELIP',
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), 'requirements.txt'))
            )
        ],
    packages=find_packages(),
)
