from setuptools import find_packages
from distutils.core import setup, Extension
import numpy as np
from Cython.Build import cythonize

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='pyarrow_ops',
    version='0.0.6',
    description='Useful data crunching tools for pyarrow',
    long_description_content_type="text/markdown",
    long_description=README,
    license='APACHE',
    packages=find_packages(),
    author='Tom Scheffers',
    author_email='tom@youngbulls.nl ',
    keywords=['arrow', 'pyarrow', 'data'],
    url='https://github.com/TomScheffers/pyarrow_ops',
    download_url='https://pypi.org/project/pyarrow-ops/'
)

install_requires = [
    'pyarrow>=3.0.0',
    'numpy>=1.16.6'
]

if __name__ == '__main__':
    print(np.get_include())

    extensions = [
        Extension("cjoin", ["pyarrow_ops/cjoin.pyx"], include_dirs=[np.get_include()])
    ]
    setup(
        **setup_args, 
        install_requires=install_requires,
        ext_modules=cythonize(extensions),
        include_dirs=np.get_include()
    )