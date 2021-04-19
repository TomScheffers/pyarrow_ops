from setuptools import setup, find_packages
from setuptools import Extension
from distutils.command.build import build as build_orig
import numpy as np
from Cython.Build import cythonize


__version__ = "0.0.7"

extensions = [
    Extension(
        name="cjoin", 
        sources=["pyarrow_ops/cjoin.c"], 
        include_dirs=[np.get_include()]
    )
]

with open('README.md') as readme_file:
    README = readme_file.read()

class build(build_orig):
    def finalize_options(self):
        super().finalize_options()
        #__builtins__.__NUMPY_SETUP__ = False
        import numpy
        for extension in self.distribution.ext_modules:
            extension.include_dirs.append(numpy.get_include())
        self.distribution.ext_modules = cythonize(self.distribution.ext_modules,
                                                  language_level=3)

setup(
    name='pyarrow_ops',
    version='0.0.7',
    description='Useful data crunching tools for pyarrow',
    long_description_content_type="text/markdown",
    long_description=README,
    license='APACHE',
    packages=find_packages(),
    author='Tom Scheffers',
    author_email='tom@youngbulls.nl ',
    keywords=['arrow', 'pyarrow', 'data'],
    url='https://github.com/TomScheffers/pyarrow_ops',
    download_url='https://pypi.org/project/pyarrow-ops/',
    include_package_data=True,
    ext_modules=extensions,
    include_dirs=np.get_include(),
    install_requires=[
        'numpy>=1.19.2',
        'PyObjC;platform_system=="Darwin"',
        'PyGObject;platform_system=="Linux"',
        'pyarrow>=3.0'
    ],
    zip_safe=False,
    
    # setup_requires=["numpy"],
    # package_data={"pyarrow_ops": ["pyarrow_ops/cjoin.pyx"]},
    # cmdclass={"build": build},
)
