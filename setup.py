from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().splitlines()

setup(
    name                = 'nutellaAgent',
    version             = '0.2.0',
    description         = 'for nutella service',
    author              = 'songpie',
    author_email        = 'songmi.ohh@gmail.com',
    license             = 'MIT',
    install_requires    = requirements,
    packages            = find_packages(exclude = []),
    keywords            = ['nutellaAgent'],
    python_requires     = '>=3.7',
    package_data        = {},
    zip_safe            = False,
    classifiers         = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)